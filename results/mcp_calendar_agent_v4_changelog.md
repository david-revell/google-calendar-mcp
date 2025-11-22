# v4_changelog.md  
Version 4 – Event Selection & Update Intelligence  
Date: 18 Nov 2025

## 1. What v4 was supposed to achieve  
Version 4 focused on one core capability:

**Update a meeting without the user providing the event_id.**

To do this, the agent needed to:

- read event IDs from `list_events` output  
- pick the correct event (e.g., “tomorrow at 11:00”)  
- avoid inventing fake IDs  
- choose the correct tool (`update_event`)  
- provide clean start/end times  

This was the entire purpose of v4.

## 2. Problems encountered

### 2.1 Model inventing fake event IDs
Examples:
- `<id>`
- `next_meeting`
- `id`

This prevented fallback logic from running.

### 2.2 Model stopped choosing update_event
Removing the example that showed update logic broke tool selection.

### 2.3 Wrong fallback date window
Fallback always queried `today → 7d`, even when user meant tomorrow.

### 2.4 Extractor never saw the actual text
`str(events_text)` gave the repr of a CallToolResult, not the plaintext.

## 3. Solutions that worked

### 3.1 Stronger prompt rules
- never invent event_id  
- leave event_id out if unknown  
- use update_event for move/change/reschedule  

### 3.2 Correct fallback date logic
Detect “tomorrow” and query exactly tomorrow-only.

### 3.3 Correct text extraction
Use:

```python
raw_text = events_text.content[0].text
```

so extractor sees the actual calendar text.

## 4. Outcome
v4 now:

- reads event IDs correctly  
- selects correct meeting  
- updates it cleanly  
- avoids placeholders  
- uses update_event reliably  
- handles “tomorrow” properly  
