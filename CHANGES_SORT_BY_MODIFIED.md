# Changes to Sort Saved Chats by Last Modified Time

## 1. Added Last Modified Timestamp

- Added a `last_modified` timestamp to each chat
- Updated the timestamp whenever a chat is modified:
  - When a user message is added
  - When an assistant message is added
  - When the model is changed
  - When a chat is loaded

## 2. Updated Saved Chats Sorting

- Changed the sorting in the saved chats view to use `last_modified` instead of `created_at`
- Added fallback to `created_at` for backward compatibility with existing chats
- Updated the display to show both "Last modified" and "Created" timestamps

## 3. Updated Chat Selection Logic

- When selecting a chat (e.g., if the current chat is invalid), now uses the most recently modified chat instead of the most recently created one

These changes ensure that the most recently used chats appear at the top of the saved chats list, making it easier to find and continue recent conversations.