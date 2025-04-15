function scrollToLatestMessage() {
    // Find all message divs based on style used in app.py
    const messageElements = document.querySelectorAll('#message-container > div[style*="margin:4px 0"]');
    const lastMessage = messageElements[messageElements.length - 1];

    if (lastMessage) {
        // Use setTimeout to allow the browser to render the new element first
        setTimeout(() => {
            // scrollIntoView on an element will scroll the window/viewport
            // if the element's container isn't the primary scroller.
            lastMessage.scrollIntoView({
                behavior: 'smooth', // Use 'auto' if 'smooth' causes issues on BlackBerry
                block: 'start'     // Align top of message with top of scroll container (window)
            });
        }, 100); // Slightly increased delay just in case
    }
    // No fallback needed, as if there are no messages, no scroll should happen.
}

document.addEventListener('DOMContentLoaded', function() {
    const container = document.getElementById('message-container');
    if (container) {
        // Initial scroll attempt on load
        scrollToLatestMessage();

        // Observe for new messages added directly to the container
        const observer = new MutationObserver((mutations) => {
            let nodesAdded = false;
            for (const mutation of mutations) {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    // Check if the added node is one of our message divs
                    mutation.addedNodes.forEach(node => {
                        if (node.nodeType === Node.ELEMENT_NODE && node.matches('div[style*="margin:4px 0"]')) {
                             nodesAdded = true;
                        }
                    });
                }
                if (nodesAdded) break;
            }
            // Only scroll if new message nodes were actually added
            if (nodesAdded) {
                scrollToLatestMessage();
            }
        });

        observer.observe(container, {
            childList: true // Observe direct children additions/removals
        });
    } else {
        console.error("Message container with ID 'message-container' not found!");
    }
});