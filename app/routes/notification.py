from flask import Blueprint,render_template, request, session, redirect, url_for, flash
import pam
import pwd
from app.config import enqueue_job
import pymongo
from decouple import config
from app.utils.logger import logger
from app.routes.auth import login_required
from app.utils.notification import *
import time
from app.utils.db import *
 
notification_bp = Blueprint('notification', __name__)

@notification_bp.route('/send_notification_route', methods=['POST'])
@login_required
def send_notification_route():
    # Check if user is admin
    if not session.get('is_admin', False):
        flash('Access denied', 'error')
        return redirect(url_for('dashboard.dashboard'))
    
    recipient = request.form.get('recipient')
    message = request.form.get('message')
    
    if not message:
        flash('Message is required', 'error')
        return redirect(url_for('notification.admin_notifications'))
    
    client, db = get_db_connection()
    try:
        if recipient == 'all':
            # Get all usernames
            users = [user['username'] for user in db.users.find({}, {'username': 1})]
            
            # Create a notification for each user
            #users=["amin"]*15
            enqueue_job(send_bulk_notification,message,users)
            # for username in users:
            #     send_notification(username,message)
            #     time.sleep(.1)

            
            flash(f'Notification sent to all users', 'success')
        else:
            # Create a notification for the specific user
            send_notification(recipient,message)
            flash(f'Notification sent to {recipient}', 'success')
            
        return redirect(url_for('notification.admin_notifications'))
    finally:
        client.close()

@notification_bp.route('/admin_notifications', methods=['GET'])
@login_required
def admin_notifications():
    # Check if user is admin
    if not session.get('is_admin', False):
        flash('Access denied', 'error')
        return redirect(url_for('dashboard.dashboard'))
    
    client, db = get_db_connection()
    try:
        # Get all users
        users = [user['username'] for user in db.users.find({}, {'username': 1})]
        
        # Get all notifications
        all_notifications = list(db.notifications.find().sort('created_at', pymongo.DESCENDING))
        
        # Get unread notifications count
        unread_count = get_unread_notifications_count(session['username'])
        
        return render_template('admin_notifications.html',
                              users=users,
                              all_notifications=all_notifications,
                              unread_notifications_count=unread_count)
    finally:
        client.close()

@notification_bp.route('/mark_notification_read', methods=['POST'])
@login_required
def mark_notification_read():
    notification_id = request.form.get('notification_id')
    
    if not notification_id:
        flash('Invalid notification', 'error')
        return redirect(url_for('notification.notifications'))
    
    client, db = get_db_connection()
    try:
        # Update the notification
        result = db.notifications.update_one(
            {'_id': ObjectId(notification_id), 'username': session['username']},
            {'$set': {'read': True}}
        )
        
        if result.modified_count > 0:
            flash('Notification marked as read', 'success')
        else:
            flash('Failed to update notification', 'error')
            
        return redirect(url_for('notification.notifications'))
    finally:
        client.close()
@notification_bp.route('/notifications')
@login_required
def notifications():
    client, db = get_db_connection()
    try:
        # Get all notifications for the current user
        notifications = list(db.notifications.find(
            {'username': session['username']}
        ).sort('created_at', pymongo.DESCENDING))
        
        # Get unread notifications count
        unread_count = get_unread_notifications_count(session['username'])
        
        return render_template('notifications.html', 
                              notifications=notifications,
                              unread_notifications_count=unread_count)
    finally:
        client.close()
        
        
@notification_bp.route('/mark_all_notifications_read', methods=['POST'])
@login_required
def mark_all_notifications_read():
    client, db = get_db_connection()
    try:
        # Update all unread notifications for the logged-in user
        result = db.notifications.update_many(
            {'username': session['username'], 'read': {'$ne': True}},
            {'$set': {'read': True}}
        )
        flash(f"Marked {result.modified_count} notifications as read.", "success")
        return redirect(url_for('notification.notifications'))
    finally:
        client.close()
