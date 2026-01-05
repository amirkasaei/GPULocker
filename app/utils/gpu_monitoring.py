import subprocess
import re
import os
import pwd
import pickle
import pymongo
import asyncio
#import schedule as sched_module
from datetime import datetime, timedelta
from app.utils.logger import logger
from app.utils import notification
from app.utils.db import MongoDBConnection
from decouple import config, Csv
from app.utils.db import get_db_connection,update_allocation_status
from datetime import datetime, timedelta
from bson import ObjectId
import json
from app.config import REDIS_BINARY,REDIS_KEYS,REDIS_CLIENT
from app.utils.redis_utils import get_available_gpus,set_available_gpus,DistributedLock
from bot import build_bot
def set_gpu_permission(username, gpu_id, grant=True):
    """Set or remove GPU permission for a user
    Args:
        username: Username to modify permissions for
        gpu_id: GPU ID to modify permissions for
        grant: True to grant access, False to remove access
    Returns:
        bool: Success status
    """
    try:
        # authorized_users = config('PRIVILEGED_USERS', cast=Csv())
        # if username in authorized_users and not grant:
        #     return True
        user_info = pwd.getpwnam(username)
        uid = user_info.pw_uid
        gpu_device = f'/dev/nvidia{gpu_id}'
        subprocess.run(['sudo', 'setfacl', '-x', f'u:{uid}', gpu_device], check=True)
        if grant:
            subprocess.run(['sudo', 'setfacl', '-m', f'u:{uid}:rw', gpu_device], check=True)
            logger.debug(f'Granted access to GPU {gpu_id} for user {username}')
        else:
            subprocess.run(['sudo', 'setfacl', '-m', f'u:{uid}:000', gpu_device], check=True)
            logger.debug(f'Removed access to GPU {gpu_id} for user {username}')
        return True
    except (KeyError, subprocess.CalledProcessError) as e:
        logger.error(f'Failed to {"grant" if grant else "remove"} access for user {username}: {str(e)}')
        return False
    


    
def check_if_available(requested_gpu_dict):
    """
    Check if requested GPU resources are available
    Args:
        requested_gpu_dict: Dictionary with gpu_type as key and requested count as value
    Returns:
        bool: True if resources are available, False otherwise
    """
    gpu_dict=get_gpu_config()
    for gpu_type, requested_count in requested_gpu_dict.items():
        # Check if gpu_type exists in GPU_DICT
        if gpu_type not in gpu_dict.keys():
            logger.error(f"Requested GPU type {gpu_type} not available")
            return False
            
        # Check if requested count is valid
        if requested_count < 0:
            logger.error(f"Invalid requested count {requested_count} for {gpu_type}")
            return False
            
        # Check if enough GPUs of this type are available
        if len(gpu_dict[gpu_type]) < requested_count:
            logger.error(f"Not enough {gpu_type} GPUs available. Requested: {requested_count}, Available: {len(gpu_dict[gpu_type])}")
            return False
    
    logger.debug("Requested GPU resources are available")
    return True

def allocate_gpu(username, gpu_type, gpu_id, days):
    """Allocate a GPU to a user with proper permission setting and database tracking
    
    Args:
        username: Username to allocate GPU to
        gpu_type: Type of GPU being allocated
        gpu_id: ID of the GPU to allocate
        days: Number of days for allocation
        
    Returns:
        tuple: (success, allocation_id or error_message)
    """
    try:
        # Remove the redundant lock since lock_gpu() already has one
        available_gpus = get_available_gpus()
        
        # Verify GPU is still available
        if gpu_id not in available_gpus.get(gpu_type, []):
            return False, "GPU no longer available"
        
        # Remove GPU from available pool
        available_gpus[gpu_type].remove(gpu_id)
        set_available_gpus(available_gpus)
        
        logger.debug(f"Allocating GPU {gpu_id} ({gpu_type}) to user {username} for {days} days")
        
        # Calculate expiration time
        allocation_time = datetime.now()
        expiration_time = allocation_time + timedelta(days=days)
        # Connect to database
        client, db = get_db_connection()
    
        try:
            # Step 1: Set GPU permission for user
            if set_gpu_permission(username, gpu_id, grant=True):
                logger.debug(f"Set permissions for GPU {gpu_id} for user {username}")
                
                try:
                    # Step 2: Record allocation in MongoDB
                    allocation_id = db.gpu_allocations.insert_one({
                        'username': username,
                        'gpu_type': gpu_type,
                        'gpu_id': gpu_id,
                        'allocated_at': allocation_time,
                        'expiration_time': expiration_time,
                        'released_at': None
                    }).inserted_id
                    
                    # Step 3: Schedule monitoring jobs for this allocation
                    schedule_allocation_monitoring(str(allocation_id), {
                        'username': username,
                        'gpu_type': gpu_type,
                        'gpu_id': gpu_id,
                        '_id': allocation_id
                    })
                    
                    logger.info(f"Granted access to GPU {gpu_id} ({gpu_type}) for user {username} until {expiration_time}")
                    return True, allocation_id
                    
                except Exception as e:
                    # Rollback Step 1: Remove permissions if database insert fails
                    logger.error(f"Failed to record allocation in database: {str(e)}")
                    set_gpu_permission(username, gpu_id, grant=False)
                    return False, f"Database error: {str(e)}"
            else:
                return False, "Failed to set GPU permissions"
        except Exception as e:
            logger.error(f"Error during GPU allocation: {str(e)}")
            return False, str(e)
        finally:
            client.close()
    except Exception as e:
        logger.error(f"Error in allocate_gpu: {str(e)}")
        return False, str(e)
    
def reset_gpu_access():
    
    try:
        # If we get here, all GPUs are free
        #logger.debug('All available GPUs are idle, resetting permissions')
        
        # Collect all GPU IDs to pass to the script
        all_gpu_ids = []
        gpu_dict=get_gpu_config()
        for gpu_type in gpu_dict.keys():
            for gpu_id in gpu_dict[gpu_type]:
                all_gpu_ids.append(str(gpu_id))
        
        # Call the setup_gpus.sh script with all GPU IDs as arguments and suppress output
        if all_gpu_ids:
            try:
                # Use absolute path or path relative to the application root
                script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'setup_gpus.sh')
                # Construct the command with all GPU IDs as arguments
                cmd = ['sudo', script_path] + all_gpu_ids
                subprocess.run(cmd, check=True, #capture_output=True,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL
                               )
                logger.info(f'Reset permissions for GPUs {all_gpu_ids} using setup_gpus.sh')
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f'Failed to reset permissions using setup_gpus.sh: {str(e)}')
                return False
        return True    
    except Exception as e:
        logger.error(f'Error in reset_gpu_access: {str(e)}')
        return False
    
    
def reset_user_access():
    """
    Reset GPU access by checking each user's access and removing it only if they
    shouldn't have access (not in active allocation within penalty period or privileged).
    
    Expects:
        PRIVILEGED_USERS in .env as comma-separated usernames
        USER_PENALTY in .env as integer (hours)
    Returns:
        bool: True if all permissions were successfully reset, False otherwise
    """
    try:
        logger.debug('Attempting to reset user-specific GPU access')
        
        # Get penalty period from .env
        penalty_hours = config('USER_LOCKOUT_HOURS', default=24, cast=int)
        authorized_users = config('PRIVILEGED_USERS', cast=Csv())
        current_time = datetime.now()
        
        # Connect to MongoDB
        client, db = get_db_connection()
        
        # First, collect all users that should keep access
        users_to_keep = set()  # Using a set to avoid duplicates
        
        # Check each GPU's allocations
        gpu_dict=get_gpu_config()
        for gpu_type in gpu_dict.keys():
            for gpu_id in gpu_dict[gpu_type]:
                # Find active allocation for this GPU
                allocation = db.gpu_allocations.find_one({
                    'gpu_id': gpu_id,
                    'gpu_type': gpu_type,
                    'released_at': None
                })
                
                if allocation:
                    # expiration_with_penalty = allocation['expiration_time'].replace(
                    #     hour=allocation['expiration_time'].hour + penalty_hours
                    # )
                    
                    # if current_time < expiration_with_penalty:
                    users_to_keep.add((allocation['username'], gpu_id))
                    available_gpus=get_available_gpus()
                    available_gpus[gpu_type].remove(gpu_id)
                    set_available_gpus(available_gpus)
                    logger.debug(f"Found active allocation for GPU {gpu_id} ({gpu_type}) for user {allocation['username']}")
                    logger.debug(f"Removing GPU {gpu_id} from available {gpu_type} GPUs")
                    # else:
                    #     # Update the allocation to mark it as released
                    #     update_allocation_status(db, allocation['_id'], released=True)
                    #     logger.info(f"Marked allocation {allocation['_id']} as released due to expiration")
        
        # Add privileged users for all GPUs
        for username in authorized_users:
            try:
                pwd.getpwnam(username)  # Verify user exists
                for gpu_type in gpu_dict.keys():
                    for gpu_id in gpu_dict[gpu_type]:
                        users_to_keep.add((username, gpu_id))
            except KeyError:
                logger.warning(f'Privileged user {username} not found in system')
        
        # Check and update access for each GPU
        for gpu_type in gpu_dict.keys():
            for gpu_id in gpu_dict[gpu_type]:
                gpu_device = f'/dev/nvidia{gpu_id}'
                try:
                    # Get current ACL entries
                    acl_output = subprocess.run(
                        ['sudo', 'getfacl', gpu_device],
                        capture_output=True,
                        text=True,
                        check=True
                    ).stdout
                    
                    # Parse ACL entries to get current users
                    current_users = []
                    for line in acl_output.splitlines():
                        if line.startswith('user:'):
                            # Format is "user:username:rw-"
                            parts = line.split(':')
                            if len(parts) >= 2 and parts[1]:  # Skip default user entry
                                current_users.append(parts[1])
                    
                    # Remove access for users who shouldn't have it
                    for username in current_users:
                        if (username, gpu_id) not in users_to_keep:
                            # Check if user is actively using the GPU before removing access
                            set_gpu_permission(username, gpu_id, grant=False)
                            logger.debug(f'Removed access to GPU {gpu_id} for user {username}')
                    
                    # Grant access to users who should have it but don't
                    for username, kept_gpu_id in users_to_keep:
                        if kept_gpu_id == gpu_id and username not in current_users:
                            set_gpu_permission(username, gpu_id, grant=True)
                            logger.debug(f'Granted access to GPU {gpu_id} for user {username}')
                
                except subprocess.CalledProcessError as e:
                    logger.error(f'Failed to manage ACL for GPU {gpu_id}: {str(e)}')
                    client.close()
                    return False
        
        logger.info('Successfully reset and configured user-specific GPU access')
        client.close()
        return True
    
    except Exception as e:
        logger.error(f'Unexpected error while resetting user access: {str(e)}')
        if 'client' in locals():
            client.close()
        return False


def check_expired_reservations():
    logger.info("Running scheduled check for expired reservations")
    try:
        client, db = get_db_connection()
        current_time = datetime.now()
        penalty_hours = config('USER_LOCKOUT_HOURS', default=24, cast=int)
        
        # Find all active allocations
        active_allocations = list(db.gpu_allocations.find({'released_at': None}))
        logger.info(f"Found {len(active_allocations)} active allocations to check")
        
        # Filter allocations where expiration_time + penalty < current_time
        allocations_to_check = []
        for allocation in active_allocations:
            expiration_time = allocation['expiration_time']
            expiration_with_penalty = expiration_time+timedelta(hours=penalty_hours)
            if current_time > expiration_with_penalty:
                logger.debug(f"Allocation {allocation['_id']} has expired (expiration: {expiration_time}, with penalty: {expiration_with_penalty})")
                allocations_to_check.append(allocation)
                
        logger.info(f"Found {len(allocations_to_check)} allocations within penalty period to check for GPU usage")
        
        for allocation in allocations_to_check:
            username = allocation['username']
            gpu_id = allocation['gpu_id']
            gpu_type = allocation['gpu_type']
            allocation_id = allocation['_id']
            
            logger.debug(f"Checking allocation {allocation_id}: GPU {gpu_id} ({gpu_type}) for user {username}")
            
            # Check if the user is actually using the GPU
            if not is_user_using_gpu(username, gpu_id) or config('FORCE_REVOKE', default=False, cast=bool):
                logger.info(f"User {username} is not using allocated GPU {gpu_id}. Releasing allocation {allocation_id}.")
                
                # Use the common unallocate function
                unallocate_gpu(username, gpu_id, gpu_type, allocation_id, db, comment=f"Released due to expiration")
            else:
                logger.debug(f"User {username} is actively using GPU {gpu_id}, skipping release")
        
        logger.info("Completed checking all active allocations")
        
    except Exception as e:
        logger.error(f"Error in check_expired_reservations: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()
def cancel_allocation_monitoring(allocation_id):
    """Cancel monitoring jobs for a GPU allocation
    
    Args:
        allocation_id: String ID of the allocation
    """
    try:
        # Get jobs from Redis
        remove_job_from_redis({"id": allocation_id, "job_function": "check_allocation_utilization"})
        remove_job_from_redis({"id": allocation_id, "job_function": "check_and_revoke_idle_allocation"})
        logger.debug(f"Cancelling jobs for allocation {allocation_id} sent to redis")
         
    except Exception as e:
        logger.error(f"Failed to cancel monitoring jobs for allocation {allocation_id}: {str(e)}")
def remove_job_from_redis(job_data):
    REDIS_BINARY.rpush(
        REDIS_KEYS['scheduler_cancel_job_queue'],
        pickle.dumps(job_data)
    )
def add_job_to_redis(job_data):
    REDIS_BINARY.rpush(
        REDIS_KEYS['scheduler_job_queue'],
        pickle.dumps(job_data)
    )

def schedule_allocation_monitoring(allocation_id, allocation_data):
    """Schedule monitoring jobs for a GPU allocation
    
    Args:
        allocation_id: String ID of the allocation
        allocation_data: Dictionary with allocation details
    """
    try:
        logger.debug(f"Scheduling monitoring jobs for allocation {allocation_id}")
        
        utilization_period_minutes = config('GPU_ACTIVITY_CHECK_MINUTES', default=5, cast=int)
        
       
        idle_check_hours=config('REVOKE_IDLE_GPU_AFTER_HOURS', default=2, cast=int)
        add_job_to_redis({
                'job_function': "check_allocation_utilization",
                'job_input': allocation_data,
                'job_interval': utilization_period_minutes,
                'job_unit': 'minutes'
            })
        add_job_to_redis({
                'job_function': "check_and_revoke_idle_allocation",
                'job_input': allocation_data,
                'job_interval': idle_check_hours,
                'job_unit': 'hours'
            })
        
        
        logger.info(f"Scheduled monitoring jobs for redis {allocation_id}")
        
        # Run the utilization check immediately to get initial data
        check_allocation_utilization(allocation_data)
        
    except Exception as e:
        logger.error(f"Failed to schedule monitoring jobs for allocation {allocation_id}: {str(e)}")
def restore_monitoring_jobs():
    """
    Restore monitoring jobs for all active allocations when the application starts
    """
    try:
        logger.info("Restoring monitoring jobs for existing allocations")
        
        # Connect to database
        with MongoDBConnection() as (client, db):
            # Find all active allocations
            active_allocations = list(db.gpu_allocations.find({'released_at': None}))
            logger.info(f"Found {len(active_allocations)} active allocations to monitor")
            
            # Schedule monitoring jobs for each allocation
            for allocation in active_allocations:
                allocation_id = str(allocation['_id'])
                allocation_data = {
                    'username': allocation['username'],
                    'gpu_type': allocation['gpu_type'],
                    'gpu_id': allocation['gpu_id'],
                    '_id': allocation['_id']
                }
                
                # Schedule monitoring jobs
                schedule_allocation_monitoring(allocation_id, allocation_data)
                check_and_revoke_idle_allocation(allocation_data)
            logger.info("Successfully restored monitoring jobs for existing allocations")
            
    except Exception as e:
        logger.error(f"Error restoring monitoring jobs: {str(e)}")

def notify_users_of_unallocation():
    """Notify users about the unallocation of a GPU"""
    try:
        # Logic to send notification (e.g., email, message, etc.)
        with MongoDBConnection() as (client, db):
            users_to_notify = list(db.gpu_notif_list.find())
            if len(users_to_notify)==0:
                logger.info("No users waiting for notification")
                return
            notification_message = "New GPU is available."
            for user in users_to_notify:
                username = user.get("username")
                if username:
                    
                    # Delete user from notification list after sending
                    db.gpu_notif_list.delete_one({'username': username})
                    logger.info(f"Deleted user {username} from notification list after sending notification.")
                    # Send notification
                    notification.send_notification(username, notification_message)
                    logger.info(f"Available GPU Notification sent to user: {username}")
                else:
                    logger.warning("Found a user entry without a username, skipping deletion.")
        
        # Log the completion of the notification process
        logger.info("All notifications sent and users deleted from the notification list.")
        
    except Exception as e:
        logger.error(f"Failed to send notification for unallocation: {str(e)}")

def unallocate_gpu(username, gpu_id, gpu_type, allocation_id, db, comment=None):
    """Release a GPU allocation with proper cleanup of permissions and database
    
    Args:
        username: Username that currently has the GPU allocated
        gpu_id: ID of the GPU to release
        gpu_type: Type of the GPU being released
        allocation_id: Database ID of the allocation
        db: MongoDB database connection
        
    Returns:
        bool: Success status
    """
    try:
        logger.debug(f"Releasing GPU {gpu_id} ({gpu_type}) from user {username}")
        
        # First, terminate any processes the user has running on this GPU
        authorized_users = config('PRIVILEGED_USERS', cast=Csv())
        if username not in authorized_users:
            try:
                # Get all processes running on this GPU
                nvidia_smi = subprocess.run(
                    ['sudo', 'nvidia-smi', '--id=' + str(gpu_id), '--query-compute-apps=pid', '--format=csv,noheader'],
                    capture_output=True, text=True, check=True
                )
                
                # For each process, check if it belongs to the user and kill it if so
                for line in nvidia_smi.stdout.strip().split('\n'):
                    if not line.strip():
                        continue
                    
                    pid = line.strip()
                    try:
                        process_owner = subprocess.run(
                            ['ps', '-o', 'user=', '-p', pid],
                            capture_output=True, text=True, check=True
                        ).stdout.strip()
                        
                        if process_owner == username:
                            logger.info(f"Terminating process {pid} owned by {username} on GPU {gpu_id}")
                            subprocess.run(['sudo', 'kill', '-9', pid], check=True)
                    except subprocess.CalledProcessError:
                        continue
                
                logger.debug(f"Terminated all processes for user {username} on GPU {gpu_id}")
            except Exception as e:
                logger.error(f"Error terminating processes on GPU {gpu_id}: {str(e)}")
                # Continue with release even if process termination fails
        
        # Cancel monitoring jobs for this allocation
        cancel_allocation_monitoring(str(allocation_id) if isinstance(allocation_id, ObjectId) else allocation_id)
        
        # Use distributed lock for the entire unallocation process
        with DistributedLock(REDIS_KEYS['gpu_lock']):
            # Get current available GPUs from Redis
            available_gpus = get_available_gpus()
            
            # Verify GPU configuration
            gpu_config = get_gpu_config()
            if not gpu_config or gpu_type not in gpu_config or gpu_id not in gpu_config[gpu_type]:
                logger.error(f"Invalid GPU configuration for {gpu_type} {gpu_id}")
                return False
            
            try:
                # Step 1: Add GPU back to available pool
                if gpu_type not in available_gpus:
                    available_gpus[gpu_type] = []
                if gpu_id not in available_gpus[gpu_type]:
                    available_gpus[gpu_type].append(gpu_id)
                    # Update available GPUs in Redis
                    set_available_gpus(available_gpus)
                logger.debug(f"Added GPU {gpu_id} back to available pool")
                
                try:
                    # Step 2: Mark allocation as released in database
                    if update_allocation_status(db, allocation_id, released=True, comment=comment):
                        logger.debug(f"Updated allocation {allocation_id} status to released")
                        
                        try:
                            # Step 3: Remove user's access to the GPU
                            if set_gpu_permission(username, gpu_id, grant=False):
                                logger.info(f"Successfully released GPU {gpu_id} from user {username}")
                                
                                # Notify users about the unallocation
                                notify_users_of_unallocation()
                                return True
                            else:
                                raise Exception(f"Failed to remove permissions for GPU {gpu_id}")
                            
                        except Exception as e:
                            # Rollback Step 2: Revert the database update
                            logger.error(f"Failed to remove permissions for GPU {gpu_id}: {str(e)}")
                            update_allocation_status(db, allocation_id, released=False)
                            # Rollback Step 1: Remove GPU from available pool
                            available_gpus[gpu_type].remove(gpu_id)
                            set_available_gpus(available_gpus)
                            logger.error(f"Rolled back allocation release due to permission error")
                            return False
                    else:
                        raise Exception(f"Failed to update allocation status in database")
                        
                except Exception as e:
                    # Rollback Step 1: Remove GPU from available pool
                    available_gpus[gpu_type].remove(gpu_id)
                    set_available_gpus(available_gpus)
                    logger.error(f"Error updating database for GPU {gpu_id}: {str(e)}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error managing available GPUs for {gpu_id}: {str(e)}")
                return False
                
    except Exception as e:
        logger.error(f"Unexpected error releasing GPU {gpu_id}: {str(e)}")
        return False
def is_user_using_gpu(username, gpu_id):
    """Check if a user is actively using a specific GPU
    Returns:
        bool: True if user is using the GPU, False otherwise
    """
    try:
        nvidia_smi = subprocess.run(
            ['sudo', 'nvidia-smi', '--id=' + str(gpu_id), '--query-compute-apps=pid', '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        
        if not nvidia_smi.stdout.strip():
            return False
            
        for line in nvidia_smi.stdout.strip().split('\n'):
            if not line.strip():
                continue
            
            pid = line.strip()
            try:
                process_owner = subprocess.run(
                    ['ps', '-o', 'user=', '-p', pid],
                    capture_output=True, text=True, check=True
                ).stdout.strip()
                
                if process_owner == username:
                    return True
            except subprocess.CalledProcessError:
                continue
                
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error checking if user {username} is using GPU {gpu_id}: {str(e)}")
        return False
    
def initialize_gpu_config():
    """Initialize GPU configuration in Redis"""
    try:
        gpu_config = eval(config('GPU_CONFIG'))
        REDIS_CLIENT.set(REDIS_KEYS['gpu_config'], json.dumps(gpu_config))
        logger.info(f"Initialized GPU configuration in Redis: {gpu_config}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize GPU config in Redis: {str(e)}")
        return False
    
def get_gpu_config():
    """
    Get GPU configuration from Redis
    Returns:
        dict: GPU configuration dictionary or empty dict if not found
    """
    try:
        gpu_config = REDIS_CLIENT.get(REDIS_KEYS['gpu_config'])
        if gpu_config:
            return json.loads(gpu_config)
        
        # If config not found in Redis, try to reinitialize
        logger.warning("GPU configuration not found in Redis, attempting to reinitialize")
        if initialize_gpu_config():
            gpu_config = REDIS_CLIENT.get(REDIS_KEYS['gpu_config'])
            if gpu_config:
                return json.loads(gpu_config)
        
        logger.error("Could not retrieve or initialize GPU configuration")
        return {}
        
    except Exception as e:
        logger.error(f"Error getting GPU config from Redis: {str(e)}")
        return {}
    
def initialize_gpu_tracking():
    """Initialize available GPU tracking from GPU_DICT"""
    try:
        with DistributedLock(REDIS_KEYS['gpu_lock']):
            gpu_dict = {gpu_type: list(gpus) for gpu_type, gpus in eval(config('GPU_CONFIG')).items()}
            set_available_gpus(gpu_dict)
            logger.info(f"Initialized available GPUs: {gpu_dict}")
            return True
    except Exception as e:
        logger.error(f"Failed to initialize GPU tracking: {str(e)}")
        return False
    
def get_gpu_status():
    """Get current GPU status information
    
    Returns:
        dict: Dictionary with GPU IDs as keys and status information as values
    """
    try:
        gpu_status = {}
        
        # Run nvidia-smi to get GPU utilization and memory usage
        nvidia_smi = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        
        # Parse the output
        for line in nvidia_smi.stdout.strip().split('\n'):
            if not line.strip():
                continue
                
            parts = [part.strip() for part in line.split(',')]
            if len(parts) >= 4:
                gpu_id = int(parts[0])

                utilization = float(parts[1])
                memory_used = float(parts[2])
                memory_total = float(parts[3])
                
                # Find which GPU type this ID belongs to
                gpu_type = None
                found=False
                gpu_dict=get_gpu_config()
                for type_name, ids in gpu_dict.items():
                    if gpu_id in ids:
                        found=True
                        gpu_type = type_name
                        break
                if found==False:
                    continue
                gpu_status[gpu_id] = {
                    'gpu_type': gpu_type,
                    'utilization': utilization,
                    'memory_used': memory_used,
                    'memory_total': memory_total,
                    'memory_percent': (memory_used / memory_total) * 100 if memory_total > 0 else 0
                }
        
        return gpu_status
    except Exception as e:
        logger.error(f"Error getting GPU status: {str(e)}")
        return {}

def check_allocation_utilization(allocation):
    """Check GPU utilization and memory usage for a specific allocation
    
    Args:
        allocation: Allocation object containing username, gpu_id, and gpu_type
        
    Returns:
        dict: Dictionary with utilization data or None if there's an error
    """
    try:
        # Extract allocation details
        username = allocation['username']
        gpu_id = allocation['gpu_id']
        gpu_type = allocation['gpu_type']
        allocation_id = allocation['_id']
        
        # Get current GPU status
        gpu_status = get_gpu_status()
        
        if gpu_id not in gpu_status:
            logger.warning(f"GPU {gpu_id} not found in status data")
            return None
        
        # Get GPU utilization and memory usage over a short period (1 second with 100ms sampling)
        # This will give multiple samples that we can average
        nvidia_smi_samples = subprocess.run(
            ['timeout', '1s','setsid', 'sudo', 'nvidia-smi', 
             f'--id={gpu_id}', '--query-gpu=utilization.gpu,memory.used', 
             '--format=csv,noheader', '-lms', '100'],
            capture_output=True, text=True, check=False,
        )
        
        # Parse the samples to calculate average utilization and memory usage
        utilization_samples = []
        memory_samples = []
        
        for line in nvidia_smi_samples.stdout.strip().split('\n'):
            if not line.strip():
                continue
                
            parts = [part.strip() for part in line.split(',')]
            if len(parts) >= 2:
                # Extract utilization percentage (remove the % sign)
                util_value = float(parts[0].replace('%', '').strip())
                utilization_samples.append(util_value)
                
                # Extract memory usage (format like "24056 MiB")
                memory_str = parts[1].strip()
                memory_value = float(re.search(r'(\d+)', memory_str).group(1))
                memory_unit = re.search(r'([A-Za-z]+)', memory_str).group(1)
                
                # Convert to MiB if needed
                if memory_unit.lower() == 'gib':
                    memory_value *= 1024
                    
                memory_samples.append(memory_value)
        
        # Calculate averages
        avg_utilization = max(utilization_samples) if utilization_samples else 0
        avg_memory_used = max(memory_samples) if memory_samples else 0
        
        # Create utilization record
        result = {
            'allocation_id': str(allocation_id),
            'username': username,
            'gpu_id': gpu_id,
            'gpu_type': gpu_type,
            'gpu_utilization': avg_utilization,
            'memory_used': avg_memory_used,
            'timestamp': datetime.now()
        }
        
        # Save utilization data to database
        try:
            with MongoDBConnection() as (client, db):
                # Create gpu_utilization collection if it doesn't exist
                if 'gpu_utilization' not in db.list_collection_names():
                    # Create time-based TTL index to automatically delete old records
                    # Default to keeping 7 days of history
                    ttl_days = config('GPU_UTILIZATION_HISTORY_DAYS', default=7, cast=int)
                    db.create_collection('gpu_utilization')
                    db.gpu_utilization.create_index('timestamp', expireAfterSeconds=ttl_days * 24 * 60 * 60)
                
                # Insert the utilization record
                db.gpu_utilization.insert_one(result)
                #logger.debug(f"Saved utilization data for GPU {gpu_id} (allocation {allocation_id})")
        except Exception as e:
            logger.error(f"Failed to save GPU utilization data: {str(e)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error checking allocation utilization: {str(e)}")
        return None
def check_and_revoke_idle_allocation(allocation):
    """Check if a GPU allocation has been idle for too long and revoke it if necessary
    
    Args:
        allocation: Allocation object containing username, gpu_id, gpu_type, and _id
        
    Returns:
        bool: True if allocation was revoked, False otherwise
    """
    try:
        username = allocation['username']
        gpu_id = allocation['gpu_id']
        gpu_type = allocation['gpu_type']
        allocation_id = allocation['_id']
        
        # Get configuration values
        idle_hours = config('REVOKE_IDLE_GPU_AFTER_HOURS', default=2, cast=int)
        min_gpu_util = config('MIN_GPU_UTILIZATION_PERCENT', default=5.0, cast=float)
        min_gpu_mem = config('MIN_GPU_MEMORY_GB', default=2.0, cast=float) * 1024  #MB
        
        # Connect to database
        with MongoDBConnection() as (client, db):
            # First check if the allocation has existed for at least REVOKE_IDLE_GPU_AFTER_HOURS
            allocation_record = db.gpu_allocations.find_one({'_id': allocation_id})
            if not allocation_record:
                logger.debug(f"Allocation {allocation_id} not found in database")
                return False
                
            # Calculate how long the allocation has existed
            allocation_time = allocation_record['allocated_at']
            current_time = datetime.now()
            allocation_age_hours = (current_time - allocation_time).total_seconds() / 3600
            # If allocation hasn't existed for the minimum time, skip the check
            if allocation_age_hours < idle_hours:
                logger.debug(f"Allocation {allocation_id} has only existed for {allocation_age_hours:.2f} hours, "
                           f"which is less than the required {idle_hours} hours for idle check")
                return False
            
            logger.debug(f"Checking if allocation {allocation_id} for GPU {gpu_id} by {username} has been idle for {idle_hours} hours")
            
            # Calculate the time threshold for utilization records
            time_threshold = current_time - timedelta(hours=idle_hours)
            
            # Find utilization records for this allocation within the time period
            utilization_records = list(db.gpu_utilization.find({
                'allocation_id': str(allocation_id),
                'timestamp': {'$gte': time_threshold}
            }).sort('timestamp', pymongo.ASCENDING))
            
            # If no records found, we can't determine if it's idle
            if not utilization_records:
                logger.debug(f"No utilization records found for allocation {allocation_id}")
                return False
                
            # Calculate maximum utilization and memory usage
            utilization_list = [record['gpu_utilization'] for record in utilization_records]
            memory_list = [record['memory_used'] for record in utilization_records]
            max_utilization=max(utilization_list)
            max_memory=max(memory_list)
            logger.debug(f"Allocation {allocation_id}: Max utilization={max_utilization}%, Max memory={max_memory}MB")
            # Check if either utilization or memory is below thresholds
            if len(set(memory_list))==1 and len(set(utilization_list))==1:
                logger.info(f"Allocation {allocation_id} for GPU {gpu_id} by {username} has been idle for at least {idle_hours} hours. "
                           f"Max utilization: {max_utilization}% (threshold: {min_gpu_util}%), "
                           f"Max memory: {max_memory}MB (threshold: {min_gpu_mem}MB)")
                
                # Revoke the allocation
                if not is_user_using_gpu(username, gpu_id) or config('FORCE_REVOKE', default=False, cast=bool):
                    logger.info(f"GPU {gpu_id} is not being used by {username}, skipping idle check")
                    if unallocate_gpu(username, gpu_id, gpu_type, allocation_id, db, comment=f"Released due to idle"):
                        logger.info(f"Successfully revoked idle allocation {allocation_id} for GPU {gpu_id} from user {username}")
                        return True
                    else:
                        logger.error(f"Failed to revoke idle allocation {allocation_id}")
                        return False
                else:
                    authorized_users = config('PRIVILEGED_USERS', cast=Csv())
                    for user in authorized_users:
                        notification.send_notification(user,f"GPU {gpu_id} is being idle at least {idle_hours} hours, but {username} has active process on it.")
                   
                    notification.send_notification(username,f"GPU {gpu_id} is being idle at least {idle_hours} hours, but you have active process on it please consider releasing it.")
                    logger.warning(f"GPU {gpu_id} is being used by {username}, but utilization and memory are below thresholds")
                    return False
            else:
                logger.debug(f"Allocation {allocation_id} is not idle (util: {max_utilization}%, mem: {max_memory}MB)")
                return False
                            
    except Exception as e:
        logger.error(f"Error checking idle allocation {allocation['_id'] if 'allocation' in locals() else 'unknown'}: {str(e)}")
        return False