
# School Multi-Agent *WIP*
This is my personal project to help me finish my last year of school. I'm not following any tutorial or using anything for inspiration, just figuring it out.

## Completed so far
### Calendar Agent (iOS Calendar API/CalDAV • Llama-4)
### Functionality:
<details>
  <summary>Create single, multiple events in a user-specified calendar</summary>
“Make an event called ‘Test Event’ in the Assignments calendar for August 15, 2025.”

“Make events called ‘Test Event’, 'Test Event2,' and 'Test Event3' in the Assignments calendar for August 15, 2025.”  
</details>
<details>
  <summary>Delete single, multiple events in a user-specified calendar</summary>
"Delete the event called 'Test Event' in the Assignments calendar for August 15, 2025."

“Delete the events named ‘Test Event’, 'Test Event2,' and 'Test Event3' in the Assignments calendar for August 15, 2025.”
</details>

https://github.com/user-attachments/assets/c547f965-47d2-458f-b9d9-27e610c3d61b

---


## Soon-to-be-completed Agentic Structure
### Notetaker Agent
**Tools**
- `notes_search` — run vector search over stored note chunks, return relevant data for downstream summarization
- `chunk_and_embed` - pull new/updated notes & class materials from Canvas API (parse.bot) daily, extract text, chunk + embed, and upsert into Delta + vector index for semantic search
- `ingest_doc` - extract text/metadata into raw store

### Planner Agent
**Tools**
- `create_events_from_syllabus` - read assignment/test dates from pdf syllabus, add to ios calendar (pass context with calendar agent)
- `update_events` - Read canvas notifications and determine if events need to be updated

### Homework Agent
**Tools**
- `completer` — assist on homework (pass context with Notetaker)

### Calendar Agent
**Tools**
- `create_calendar_event(s)` — add events to Apple Calendar  
- `delete_calendar_event(s)` — delete events from Apple Calendar  
- `list_calendars` — list all user's calendars
