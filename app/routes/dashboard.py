from app.config import enqueue_job
from flask import Blueprint,render_template,Response, request, session, redirect, url_for, flash
from decouple import config,Csv
from app.utils.logger import logger
import jdatetime
import pymongo
from app.utils.gpu_monitoring import get_gpu_status,unallocate_gpu,get_gpu_config,set_available_gpus,set_gpu_permission,allocate_gpu
from app.utils.db import *
from app.utils.redis_utils import DistributedLock
from app.config import REDIS_CLIENT,REDIS_KEYS,DISK_CACHE_TIMEOUT
from app.utils.disk import get_disk_cache,get_user_disk_usage,set_disk_cache,update_user_disk_cache
from app.routes.auth import login_required
import json
from datetime import datetime
from app.utils.gpu_monitoring import get_available_gpus
from app.utils.notification import get_unread_notifications_count
dashboard_bp = Blueprint('dashboard', __name__,static_url_path="dashboard")
@dashboard_bp.route('/')
def index():
    """Root route that redirects to the login page"""
    return redirect(url_for('auth.login'))
@dashboard_bp.route('/schedule')
@login_required
def schedule():
    try:
        with MongoDBConnection() as (client, db):
            # Query for all active allocations
            allocations = list(db.gpu_allocations.find({
                'released_at': None
            }).sort('expiration_time', pymongo.ASCENDING))  # Sort by expiration time
            
            # Format dates for display
            formatted_allocations = format_allocations_for_display(allocations)
            
    except Exception as e:
        logger.error(f"Error fetching schedule: {str(e)}")
        formatted_allocations = []
    
    # Get initial GPU status for first page load
    gpu_status = get_gpu_status()
    
    # Get refresh rate from .env
    refresh_rate = config('GPUs_STATUS_REFRESH_RATE_SECONDS', default=5, cast=float)
    refresh_rate_ms = int(refresh_rate * 1000)  # Convert to milliseconds
    
    # Get unread notifications count
    unread_count = get_unread_notifications_count(session['username'])
    
    return render_template('schedule.html', 
                          allocations=formatted_allocations,
                          gpu_status=gpu_status,
                          refresh_rate_ms=refresh_rate_ms,
                          unread_notifications_count=unread_count)

def tail(file, n):
    import os
    with open(file, 'rb') as f:
        f.seek(0, os.SEEK_END)
        buffer = bytearray()
        pointer = f.tell()
        while pointer >= 0 and n > 0:
            f.seek(pointer)
            pointer -= 1
            new_byte = f.read(1)
            if new_byte == b'\n':
                n -= 1
            buffer.extend(new_byte)
        return buffer[::-1].decode('utf-8')

@dashboard_bp.route('/logs')
@login_required
def get_logs():
    LOG_FILE_PATH="/home/user01/GPULocker/gpulock.log"
    try:
        lines = int(request.args.get('lines', 200))  # Default to 200 lines
        lines = max(1, min(300, lines))  # Limit between 1 and 300
        logs = tail(LOG_FILE_PATH, lines)
        return Response(logs, mimetype='text/plain')
    except Exception as e:
        return f"Error reading log file: {str(e)}", 500


@dashboard_bp.route('/dashboard')
@login_required
def dashboard():
    username = session["username"]
    
    # Get page parameters
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Number of items per page
    
    try:
        with MongoDBConnection() as (client, db):
            # Query for user's active allocations first
            active_allocations = list(db.gpu_allocations.find({
                'username': username,
                'released_at': None
            }).sort('expiration_time', pymongo.DESCENDING))
            
            # Query for user's allocation history
            history_allocations = list(db.gpu_allocations.find({
                'username': username,
                'released_at': {'$ne': None}
            }).sort('released_at', pymongo.DESCENDING))
            
            # Combine active and history allocations
            allocations = active_allocations + history_allocations
            
            # Calculate pagination
            total_allocations = len(allocations)
            total_pages = max(1, (total_allocations + per_page - 1) // per_page)
            page = min(page, total_pages)  # Ensure page is within bounds
            
            # Slice the allocations for current page
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            current_allocations = allocations[start_idx:end_idx] if allocations else []
            
            # Format dates for display
            current_allocations = format_allocations_for_display(current_allocations)
            
            # For admin users, get all allocations with pagination
            all_allocations = []
            admin_page = request.args.get('admin_page', 1, type=int)
            authorized_users = config('PRIVILEGED_USERS', cast=Csv())
            is_admin = username in authorized_users
            total_admin_pages = 1  # Default value
            
            if is_admin:
                # Get active allocations first
                admin_active = list(db.gpu_allocations.find({
                    'released_at': None
                }).sort('expiration_time', pymongo.DESCENDING))
                
                # Get allocation history
                admin_history = list(db.gpu_allocations.find({
                    'released_at': {'$ne': None}
                }).sort('released_at', pymongo.DESCENDING))
                
                all_allocations = admin_active + admin_history
                
                # Calculate admin pagination
                total_admin_allocations = len(all_allocations)
                total_admin_pages = max(1, (total_admin_allocations + per_page - 1) // per_page)
                admin_page = min(admin_page, total_admin_pages)  # Ensure admin_page is within bounds
                
                # Slice admin allocations for current page
                admin_start = (admin_page - 1) * per_page
                admin_end = admin_start + per_page
                all_allocations = all_allocations[admin_start:admin_end] if all_allocations else []
                all_allocations = format_allocations_for_display(all_allocations)
            
            # Get all users for admin dropdown
            if is_admin:
                all_users = [user['username'] for user in db.users.find({}, {'username': 1})]
            else:
                all_users = []
            
            # Get available GPUs from Redis
            available_gpus = get_available_gpus()
            
            # Get user's disk usage from Redis cache
            disk_usage_data = get_disk_cache(username)
            if disk_usage_data is None:
                # Cache miss - calculate and cache
                enqueue_job(update_user_disk_cache, username)
                disk_usage_data = {
                    'used': "Loading...",
                    'used_by_others': "Loading...",
                    'free': "Loading...",
                    'total': "Loading...",
                    'percent_used': 0,
                    'percent_others': 0,
                    'percent_free': 0
                }
                # Cache the result
                #set_disk_cache(username, disk_usage_data, timeout=DISK_CACHE_TIMEOUT)
            
            # Get unread notifications count
            unread_count = get_unread_notifications_count(username)
            
            # Get GPU status from Redis or calculate
            gpu_status_key = REDIS_KEYS['gpu_status']
            gpu_status = REDIS_CLIENT.get(gpu_status_key)
            if gpu_status:
                gpu_status = json.loads(gpu_status)
            else:
                gpu_status = get_gpu_status()
                # Cache GPU status for a short time (e.g., 5 seconds)
                REDIS_CLIENT.setex(gpu_status_key, 5, json.dumps(gpu_status))
            
            # Check if user is registered for notifications
            user_notif_entry = db.gpu_notif_list.find_one({'username': username})
            is_registered_for_notifications = user_notif_entry is not None
            return render_template('dashboard.html',
                                username=username,
                                gpu_dict=available_gpus,
                                allocations=current_allocations,
                                all_allocations=all_allocations,
                                is_admin=is_admin,
                                page=page,
                                total_pages=total_pages,
                                admin_page=admin_page,
                                total_admin_pages=total_admin_pages,
                                disk_usage=disk_usage_data,
                                unread_notifications_count=unread_count,
                                gpu_status=gpu_status,
                                now=datetime.now(),
                                is_registered_for_notifications=is_registered_for_notifications,
                                all_users=all_users)
            
    except Exception as e:
        logger.error(f"Error in dashboard route: {str(e)}")
        flash("An error occurred while loading the dashboard", "error")
        return redirect(url_for('auth.login'))

@dashboard_bp.route('/release_gpu', methods=['POST'])
@login_required
def release_gpu():
    username = session['username']
    allocation_id = request.form.get('allocation_id')
    authorized_users = config('PRIVILEGED_USERS', cast=Csv())
    if username in authorized_users:
        is_admin=True 
    else:
        is_admin=False
    
    try:
        client, db = get_db_connection()
        
        # Find the allocation - if admin, don't check username
        query = {'_id': ObjectId(allocation_id), 'released_at': None}
        if not is_admin:
            # Regular users can only release their own GPUs
            query['username'] = username
            
        allocation = db.gpu_allocations.find_one(query)
        
        if not allocation:
            flash('Invalid GPU allocation or unauthorized access', 'error')
            return redirect(url_for('dashboard.dashboard'))
        
        gpu_type = allocation['gpu_type']
        gpu_id = allocation['gpu_id']
        user_username = allocation['username']  # The actual owner of the GPU
        
        # Add a comment based on who is releasing the GPU
        comment = f"Manually released by {username}"
        if is_admin and username != user_username:
            comment = f"Released by admin :{username}"
        
        # Use the common unallocate function with the comment
        if unallocate_gpu(user_username, gpu_id, gpu_type, allocation_id, db, comment=comment):
            if is_admin and username != user_username:
                # Admin releasing GPU on behalf of a user
                flash(
                    f'Successfully released GPU {gpu_id} from user {user_username}. '
                    'A 30-minute cooldown has been applied before the user can reserve again.',
                    'success'
                )
                logger.info(f"Admin {username} released GPU {gpu_id} from user {user_username}")

                # Apply cooldown for the actual GPU owner
                db.users.update_one(
                    {'username': user_username},
                    {'$set': {'last_manual_release': datetime.utcnow()}}
                )
                logger.info(f"Cooldown applied for user {user_username} after admin release")

                # ✅ Notify the user
                send_notification(
                    user_username,
                    f"Your GPU (ID {gpu_id}, type {gpu_type}) was released by an admin. "
                    "You will not be able to reserve another GPU for 30 minutes."
                )

            else:
                # Normal user releasing their own GPU
                flash(
                    f'Successfully released GPU {gpu_id} from user {user_username}. '
                    'A 30-minute cooldown has been applied before the user can reserve again.',
                    'success'
                )
                logger.info(f"Admin {username} released GPU {gpu_id} from user {user_username}")

                # Apply cooldown
                db.users.update_one(
                    {'username': username},
                    {'$set': {'last_manual_release': datetime.utcnow()}}
                )
                logger.info(f"Cooldown applied for user {username} after manual release")

                # ✅ Notify the user
                send_notification(
                    username,
                    f"You manually released GPU (ID {gpu_id}, type {gpu_type}). "
                    "You will not be able to reserve another GPU for 30 minutes."
                )
        else:
            flash('Failed to release GPU', 'error')

                
    except Exception as e:
        logger.error(f"Error in release_gpu route: {str(e)}")
        flash('Failed to release GPU', 'error')
    finally:
        if 'client' in locals():
            client.close()
    
    return redirect(url_for('dashboard.dashboard'))

@dashboard_bp.route('/extend_gpu', methods=['POST'])
@login_required
def extend_gpu():
    username = session['username']
    allocation_id = request.form.get('allocation_id')
    extension_days = request.form.get('extension_days')
    
    # Validate input
    if not allocation_id or not extension_days:
        flash('Invalid request parameters', 'error')
        return redirect(url_for('dashboard.dashboard'))
    
    try:
        extension_days = int(extension_days)
        if extension_days < 1 or extension_days > 7:
            flash('Extension days must be between 1 and 7', 'error')
            return redirect(url_for('dashboard.dashboard'))
    except ValueError:
        flash('Invalid extension days value', 'error')
        return redirect(url_for('dashboard.dashboard'))
    
    # Check admin status from session
    is_admin = session.get('is_admin', False)
    
    try:
        with MongoDBConnection() as (client, db):
            # Find the allocation - if admin, don't check username
            query = {'_id': ObjectId(allocation_id), 'released_at': None}
            if not is_admin:
                # Regular users can only extend their own GPUs
                query['username'] = username
                
            allocation = db.gpu_allocations.find_one(query)
            
            if not allocation:
                flash('Invalid GPU allocation or unauthorized access', 'error')
                return redirect(url_for('dashboard.dashboard'))
            
            # Calculate new expiration time
            from datetime import timedelta
            current_expiration = allocation['expiration_time']
            new_expiration = current_expiration + timedelta(days=extension_days)
            if datetime.now() < current_expiration:
                flash('GPU is not expired yet', 'error')
                logger.warning(f"User {username} tried to extend GPU {allocation['gpu_id']} but it is not expired yet")
                return redirect(url_for('dashboard.dashboard'))
            # Update the allocation
            result = db.gpu_allocations.update_one(
                {'_id': ObjectId(allocation_id)},
                {'$set': {'expiration_time': new_expiration}}
            )
            
            if result.modified_count > 0:
                gpu_id = allocation['gpu_id']
                gpu_type = allocation['gpu_type']
                user_username = allocation['username']
                
                if is_admin and username != user_username:
                    flash(f'Successfully extended GPU {gpu_id} for user {user_username} by {extension_days} days', 'success')
                    logger.info(f"Admin {username} extended GPU {gpu_id} for user {user_username} by {extension_days} days")
                else:
                    flash(f'Successfully extended GPU {gpu_id} by {extension_days} days', 'success')
                    logger.info(f"User {username} extended GPU {gpu_id} by {extension_days} days")
            else:
                flash('Failed to extend GPU allocation', 'error')
                
    except Exception as e:
        logger.error(f"Error in extend_gpu route: {str(e)}")
        flash('Failed to extend GPU allocation', 'error')
    
    return redirect(url_for('dashboard.dashboard'))

@dashboard_bp.route('/lock_gpu', methods=['POST'])
@login_required
def lock_gpu():
    requested_gpu_dict = {}
    requested_days = {}
    username = session['username']
    
    # --- cooldown check (30 minutes after manual release) ---
    try:
        with MongoDBConnection() as (client, db):
            user_entry = db.users.find_one({'username': username}, {'last_manual_release': 1})
            if user_entry and user_entry.get('last_manual_release'):
                from datetime import timedelta
                last_release = user_entry['last_manual_release']
                now_utc = datetime.utcnow()
                if (now_utc - last_release) < timedelta(minutes=30):
                    remaining = 30 - int((now_utc - last_release).total_seconds() / 60)
                    flash(f"You must wait {remaining} minute(s) before reserving a new GPU after a manual release.", "error")
                    # ✅ Send notification to remind them
                    send_notification(
                        username,
                        f"You attempted to reserve a GPU, but you are still under cooldown for another {remaining} minute(s). "
                        "Please wait until the 30-minute cooldown expires."
                    )
                    return redirect(url_for('dashboard.dashboard'))
    except Exception as e:
        logger.warning(f"Cooldown check failed: {e}")

    
    # Check if user is admin
    authorized_users = config('PRIVILEGED_USERS', cast=Csv())
    is_admin = username in authorized_users
    
    # Get the target user (for admin allocations)
    target_user = request.form.get('target_user', username)
    
    # If not admin, ensure target_user is the current user
    if not is_admin:
        target_user = username
    
    try:
        # Get GPU configuration from Redis
        gpu_config = get_gpu_config()
        if not gpu_config:
            logger.error("Could not retrieve GPU configuration")
            flash("System configuration error", "error")
            return redirect(url_for('dashboard.dashboard'))
        
        # Parse requested GPUs and days
        for gpu_type in gpu_config.keys():
            try:
                quantity = request.form.get(f'quantity_{gpu_type}', '0')
                days = request.form.get(f'days_{gpu_type}', '0')
                
                # Validate input
                if not quantity.isdigit() or not days.isdigit():
                    flash("Invalid input values", "error")
                    return redirect(url_for('dashboard.dashboard'))
                
                requested_gpu_dict[gpu_type] = int(quantity)
                requested_days[gpu_type] = int(days)
                
                # Validate days range
                if requested_days[gpu_type] <= 0 or requested_days[gpu_type] > 7:
                    flash("Number of days must be between 1 and 7", "error")
                    return redirect(url_for('dashboard.dashboard'))
                
            except ValueError as e:
                logger.error(f"Invalid form data: {str(e)}")
                flash("Invalid input values", "error")
                return redirect(url_for('dashboard.dashboard'))
        
        logger.info(f"User {username} requested GPUs: {requested_gpu_dict} for days: {requested_days}")
        
        # Use distributed lock for the entire allocation process
        with DistributedLock(REDIS_KEYS['gpu_lock']):
            # Get current available GPUs from Redis
            available_gpus = get_available_gpus()
            if not available_gpus:
                flash("No GPUs available at the moment", "error")
                return redirect(url_for('dashboard.dashboard'))
            
            # Validate against GPU configuration
            for gpu_type, count in requested_gpu_dict.items():
                if count > 0:
                    # Check if GPU type exists
                    if gpu_type not in gpu_config:
                        flash(f"Invalid GPU type: {gpu_type}", "error")
                        return redirect(url_for('dashboard.dashboard'))
                    
                    # Check if enough GPUs are available
                    if gpu_type not in available_gpus or len(available_gpus[gpu_type]) < count:
                        flash(f"Not enough {gpu_type} GPUs available", "error")
                        return redirect(url_for('dashboard.dashboard'))
            
            # Track successful allocations for rollback
            successful_allocations = []
            allocated_gpus = {}
            
            try:
                with MongoDBConnection() as (client, db):
                    # Perform allocations
                    for gpu_type, count in requested_gpu_dict.items():
                        if count > 0:
                            allocated_gpus[gpu_type] = []
                            
                            for _ in range(count):
                                if not available_gpus[gpu_type]:
                                    raise Exception(f"No more {gpu_type} GPUs available")
                                
                                gpu_id = available_gpus[gpu_type].pop(0)
                                
                                # Verify GPU ID is valid according to configuration
                                if gpu_id not in gpu_config[gpu_type]:
                                    raise Exception(f"Invalid GPU ID {gpu_id} for type {gpu_type}")
                                
                                # Allocate GPU to user
                                success, result = allocate_gpu(
                                    target_user,
                                    gpu_type,
                                    gpu_id,
                                    requested_days[gpu_type]
                                )
                                
                                if success:
                                    successful_allocations.append({
                                        'gpu_type': gpu_type,
                                        'gpu_id': gpu_id,
                                        'allocation_id': result
                                    })
                                    allocated_gpus[gpu_type].append(gpu_id)
                                else:
                                    # Put GPU back in available pool
                                    available_gpus[gpu_type].append(gpu_id)
                                    raise Exception(f"Failed to allocate GPU {gpu_id}: {result}")
                    
                    # Update available GPUs in Redis
                    set_available_gpus(available_gpus)
                    
                    if allocated_gpus:
                        if is_admin and target_user != username:
                            logger.info(f"Admin {username} allocated GPU {gpu_id} to user {target_user}")
                            flash(f"Successfully allocated GPUs: {allocated_gpus} to user {target_user} with expiration times: {requested_days} days", "success")
                        else:
                            flash(f"Successfully allocated GPUs: {allocated_gpus} with expiration times: {requested_days} days", "success")
                    else:
                        flash("No GPUs were allocated", "info")
                    
                    return redirect(url_for('dashboard.dashboard'))
                    
            except Exception as e:
                # Rollback all successful allocations
                logger.error(f"Error during GPU allocation: {str(e)}")
                
                with MongoDBConnection() as (client, db):
                    for alloc in successful_allocations:
                        try:
                            # Remove GPU access
                            set_gpu_permission(target_user, alloc['gpu_id'], grant=False)
                            
                            # Remove database entry
                            db.gpu_allocations.delete_one({'_id': alloc['allocation_id']})
                            
                            # Return GPU to available pool
                            available_gpus[alloc['gpu_type']].append(alloc['gpu_id'])
                            
                            logger.debug(f"Rolled back allocation for GPU {alloc['gpu_id']}")
                        except Exception as rollback_error:
                            logger.error(f"Error during rollback: {str(rollback_error)}")
                    
                    # Update available GPUs in Redis after rollback
                    set_available_gpus(available_gpus)
                
                flash(f"Failed to allocate GPUs: {str(e)}", "error")
                return redirect(url_for('dashboard.dashboard'))
                
    except Exception as e:
        logger.error(f"Unexpected error in lock_gpu: {str(e)}")
        flash("An unexpected error occurred", "error")
        return redirect(url_for('dashboard.dashboard'))

@dashboard_bp.route('/notify_user', methods=['POST'])
@login_required
def notify_user():
    from flask import jsonify
    username = session['username']
    
    try:
        with MongoDBConnection() as (client, db):
            existing_entry = db.gpu_notif_list.find_one({'username': username})
                
            if not existing_entry:
                db.gpu_notif_list.insert_one({'username': username})
                logger.info(f"User '{username}' successfully registered for GPU availability notifications.")
                return jsonify({"message": "You will be notified when GPUs are available.", "status": "success"})
            else:
                logger.warning(f"User '{username}' attempted to register for notifications multiple times.")
                return jsonify({"message": "You are already registered for notifications.", "status": "info"})
    except Exception as e:
        logger.error(f"Error adding user to notification list: {str(e)}")
        return jsonify({"message": "Failed to register for notifications.", "status": "error"}), 500

def format_allocations_for_display(allocations):
    """Format allocation dates for display using jdatetime or regular datetime based on config"""
    # Get configuration for date format from .env
    use_jalali = config('USE_JALALI_DATES', default=True, cast=bool)
    
    for allocation in allocations:
        if use_jalali:
            # Use jdatetime (Jalali/Persian calendar)
            allocation['allocated_at_str'] = jdatetime.datetime.fromgregorian(
                datetime=allocation['allocated_at']).strftime('%Y-%m-%d %H:%M:%S')
            allocation['expiration_time_str'] = jdatetime.datetime.fromgregorian(
                datetime=allocation['expiration_time']).strftime('%Y-%m-%d %H:%M:%S')
            if allocation.get('released_at'):
                allocation['released_at_str'] = jdatetime.datetime.fromgregorian(
                    datetime=allocation['released_at']).strftime('%Y-%m-%d %H:%M:%S')
        else:
            # Use regular datetime (Gregorian calendar)
            allocation['allocated_at_str'] = allocation['allocated_at'].strftime('%Y-%m-%d %H:%M:%S')
            allocation['expiration_time_str'] = allocation['expiration_time'].strftime('%Y-%m-%d %H:%M:%S')
            if allocation.get('released_at'):
                allocation['released_at_str'] = allocation['released_at'].strftime('%Y-%m-%d %H:%M:%S')
                
        if allocation.get('comment'):
            allocation['comment_str'] = allocation['comment']
        else:
            allocation['comment_str'] = '-'    
    return allocations

