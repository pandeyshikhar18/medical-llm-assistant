css = '''
<style>
/* Chat container */
.chat-message {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 1rem;
    margin: 0.8rem 0;
    border-radius: 1rem;
    max-width: 85%;
    animation: fadeIn 0.3s ease-in-out;
}

/* User message (aligned right) */
.chat-message.user {
    margin-left: auto;
    background: linear-gradient(135deg, #3a3f51, #2b313e);
    border: 1px solid #3d4456;
    justify-content: flex-end;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
}

/* Bot message (aligned left) */
.chat-message.bot {
    margin-right: auto;
    background: linear-gradient(135deg, #475063, #3b4252);
    border: 1px solid #444c5e;
    justify-content: flex-start;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}

/* Avatar styling */
.chat-message .avatar {
    flex-shrink: 0;
}
.chat-message .avatar img {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #555;
    box-shadow: 0 0 4px rgba(0,0,0,0.3);
}

/* Message text */
.chat-message .message {
    font-size: 0.95rem;
    line-height: 1.5;
    color: #f5f5f5;
    padding: 0.75rem 1rem;
    border-radius: 0.8rem;
    word-wrap: break-word;
    max-width: 600px;
}

/* Fade-in animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
'''

# Bot bubble (assistant)
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

# User bubble
user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/847/847969.png">
    </div>
</div>
'''
