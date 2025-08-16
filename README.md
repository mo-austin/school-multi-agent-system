
# School Multi-Agent *WIP*
This is my personal project to help me finish my last year of school. I'm not following any tutorial or using anything for inspiration, just figuring it out.

## Completed so far
## Calendar Agent (Model: Llama-4 • iOS Calendar API/CalDAV)

https://github.com/user-attachments/assets/5c307bc6-7e46-4fce-a5ed-a03aa18d5218

*Create single, multiple events in a user-specified calendar.*
- “Make an event called ‘Test Event’ in the Assignments calendar for August 15, 2025.”  
- “Make events called ‘Test Event’, 'Test Event2,' and 'Test Event3' in the Assignments calendar for August 15, 2025.”  

*Delete single, multiple events in a user-specified calendar.*
- "Delete the event called 'Test Event' in the Assignments calendar for August 15, 2025."
- “Delete the events named ‘Test Event’, 'Test Event2,' and 'Test Event3' in the Assignments calendar for August 15, 2025.”

---


## Soon-to-be Complete Agentic Structure

### Calendar Agent
**Tools**
- `create_calendar_event(s)` — add events to Apple Calendar  
- `delete_calendar_event(s)` — delete events from Apple Calendar  
- `list_calendars` — list all user's calendars

### Planner Agent
**Tools**
- `create_events_from_syllabus` - read assignment/test dates from pdf syllabus, add to ios calendar (pass context with calendar agent)

### Notetaker Agent
**Tools**
- `summarizer` — summarize text/PDF input and write to Delta  
- `notes_search` — find notes by class, date, or query

### Homework Agent
**Tools**
- `completer` — assist on homework (pass context with Notetaker)
