import functools
import os
import uuid
from typing import Any, Generator, Literal, Optional
from datetime import datetime, timedelta, date, time
import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
)
# from databricks_langchain.genie import GenieAgent  #Genie
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from pydantic import BaseModel

# === Calendar deps ===
from caldav import DAVClient
from icalendar import Calendar, Event
import pytz

# LLM endpoint (replace if needed)
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)


os.environ["ICLOUD_USER"] = "Austin287908@gmail.com"
os.environ["ICLOUD_APP_PW"] = "aaxi-ntga-fpwg-gpix"

DEFAULT_CALENDAR_NAME = os.getenv("DEFAULT_CAL_NAME", "Assignments")

############################################
# Apple Calendar utilities (CalDAV + ICS)

def _get_icloud_creds():
    user = os.getenv("ICLOUD_USER")
    pw = os.getenv("ICLOUD_APP_PW")
    if user and pw:
        return user, pw
    # try dbutils only if it exists (not in serving)
    try:
        import builtins
        dbu = getattr(builtins, "dbutils", None)
        if dbu is not None:
            user = dbu.secrets.get("apple", "ICLOUD_USER")
            pw   = dbu.secrets.get("apple", "ICLOUD_APP_PW")
            os.environ["ICLOUD_USER"] = user
            os.environ["ICLOUD_APP_PW"] = pw
            return user, pw
    except Exception:
        pass
    raise RuntimeError("Set ICLOUD_USER and ICLOUD_APP_PW as env vars (or load from secrets in a notebook) before running.")



def _get_calendar_by_name(name: Optional[str] = None):
    user, pw = _get_icloud_creds()
    client = DAVClient(url="https://caldav.icloud.com", username=user, password=pw)
    principal = client.principal()
    calendars = principal.calendars()
    if not calendars:
        raise RuntimeError("No iCloud calendars found for this account.")
    if name:
        for c in calendars:
            try:
                props = c.get_properties(["{DAV:}displayname"])  # best-effort
                display = props.get("{DAV:}displayname", "").strip() if props else ""
            except Exception:
                display = ""
            if display == name or getattr(c, "name", None) == name:
                return c
    # fallback to first calendar
    return calendars[0]


def _build_ics_event(summary: str, start_dt: datetime, end_dt: datetime, description: Optional[str], tz: str):
    zone = pytz.timezone(tz)
    cal = Calendar()
    cal.add("prodid", "-//school-assistant//langgraph//EN")
    cal.add("version", "2.0")
    evt = Event()
    evt.add("uid", str(uuid.uuid4()))
    evt.add("summary", summary)
    evt.add("dtstart", zone.localize(start_dt))
    evt.add("dtend", zone.localize(end_dt))
    if description:
        evt.add("description", description)
    cal.add_component(evt)
    return cal.to_ical(), str(evt["uid"])  # (bytes, uid)


############################################
# LangChain tools for the Assignment agent
############################################

def _safe_str(v):
    try:
        return str(v)
    except Exception:
        return ""

@tool("list_calendars")
def list_calendars_tool() -> str:
    """Return iCloud calendars as JSON: [{displayName, url}]. Ensures URL is a string."""
    import json
    user, pw = _get_icloud_creds()
    client = DAVClient(url="https://caldav.icloud.com", username=user, password=pw)
    principal = client.principal()
    calendars = principal.calendars() or []

    rows = []
    for c in calendars:
        try:
            props = c.get_properties(["{DAV:}displayname"])  # may fail
            display = props.get("{DAV:}displayname", "").strip() if props else ""
        except Exception:
            display = ""
        url_val = getattr(c, "url", "")  # may be a URL object
        rows.append({
            "displayName": _safe_str(display or getattr(c, "name", "")).strip(),
            "url": _safe_str(url_val),
        })

    return json.dumps(rows, ensure_ascii=False)


@tool("create_calendar_event")
def create_calendar_event_tool(
    summary: str,
    start_iso: Optional[str] = None,
    end_iso: Optional[str] = None,
    date_iso: Optional[str] = None,
    description: Optional[str] = None,
    calendar_name: Optional[str] = None,
    timezone: str = "America/New_York",
) -> str:
    """Create an Apple Calendar event via CalDAV.
    Args:
      summary: Title of the event (e.g., "CS310 HW1 Due").
      start_iso: Start datetime in ISO 8601 (e.g., "2025-09-15T23:00:00") OR date-only ("2025-09-15").
      end_iso: End datetime in ISO 8601 OR date-only. If omitted and only a date is provided, defaults to 23:59 that date.
      date_iso: Convenience date (YYYY-MM-DD). If provided, used when start/end are omitted.
      description: Optional notes (course, links).
      calendar_name: Optional iCloud calendar display name to target. Defaults to first calendar.
      timezone: IANA tz name, default America/New_York.
    Behavior:
      - If only a date is supplied (via start_iso as date-only OR date_iso), schedules **22:59 → 23:59** local time.
      - If times are supplied, uses them as-is.
    Returns:
      UID string for the created event.
    """
    cal = _get_calendar_by_name(calendar_name)

    def _parse(s: Optional[str], is_start: bool) -> Optional[datetime]:
        if not s:
            return None
        if "T" in s:
            return datetime.fromisoformat(s)
        # date-only
        d = date.fromisoformat(s)
        t = time(22, 59) if is_start else time(23, 59)
        return datetime.combine(d, t)

    # Prefer explicit start/end; fallback to date_iso
    start_candidate = start_iso or date_iso
    end_candidate = end_iso or (start_candidate if start_candidate else None)

    start_dt = _parse(start_candidate, True)
    if start_dt is None:
        raise ValueError("Provide at least start_iso (date or datetime) or date_iso.")
    end_dt = _parse(end_candidate, False)

    ics_bytes, uid = _build_ics_event(summary, start_dt, end_dt, description, timezone)
    cal.add_event(ics_bytes)
    return uid


from typing import List
from langchain_core.prompts import ChatPromptTemplate
import json
from pydantic import BaseModel

# === Planner schemas ===
class EventSpec(BaseModel):
    summary: str
    date_iso: Optional[str] = None
    start_iso: Optional[str] = None
    end_iso: Optional[str] = None
    calendar_name: str
    description: Optional[str] = None

class Plan(BaseModel):
    events: List[EventSpec]
@tool("create_multiple_calendar_events")
def create_multiple_calendar_events_tool(
    instruction_text: str,
    default_calendar_name: str,
) -> str:
    """
    Parse the user's instruction and create one or more events.
    Always list calendars first, resolve each event's calendar_name, then create events.
    Returns JSON of created events with UIDs.
    """
    import json
    from langchain_core.prompts import ChatPromptTemplate

    # Planner
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Extract calendar events from the user's text. "
         "Return a JSON object with a top-level key named events. "
         "Each event must include: summary and calendar_name. "
         "Optionally include date_iso (YYYY-MM-DD) or start_iso and end_iso, plus description. "
         "Use YYYY-MM-DD for date_iso when no time is given. Do not invent dates."),
        ("user", "{text}")
    ])
    planner = prompt | llm.with_structured_output(Plan)
    plan = planner.invoke({"text": instruction_text})

    # 2a) Fetch calendars and build lookup helpers
    calendars_json = list_calendars_tool.invoke({})
    try:
        calendars = json.loads(calendars_json)
    except Exception:
        calendars = []

    def resolve_calendar_name(requested: Optional[str], fallback: str) -> str:
        """Case-insensitive exact, then substring match on displayName. Fallback to default."""
        if not calendars:
            return requested or fallback
        target = (requested or fallback or "").casefold()
        names = [c.get("displayName", "") for c in calendars]
        names_fold = [n.casefold() for n in names]
        # exact
        if target in names_fold:
            return names[names_fold.index(target)]
        # substring
        for i, nf in enumerate(names_fold):
            if target and target in nf:
                return names[i]
        return fallback

    created = []
    for ev in plan.events:
        # pick calendar name
        cal_name = resolve_calendar_name(ev.calendar_name, default_calendar_name)

        payload = {"summary": ev.summary, "calendar_name": cal_name}
        if ev.start_iso or ev.end_iso:
            payload["start_iso"] = ev.start_iso or ev.date_iso
            payload["end_iso"] = ev.end_iso or ev.date_iso
        else:
            # your tool defaults to 22:59–23:59 for date-only
            payload["start_iso"] = ev.date_iso
        if ev.description:
            payload["description"] = ev.description

        uid = create_calendar_event_tool.invoke(payload)
        created.append({"summary": ev.summary, "calendar_name": cal_name, "uid": uid})

    return json.dumps({"created": created})


@tool("delete_calendar_event")
def delete_calendar_event_tool(
    summary: str,
    calendar_name: Optional[str] = None,
    starts_after_iso: Optional[str] = None,
    ends_before_iso: Optional[str] = None,
    max_delete: int = 4,
    case_sensitive: bool = False,
    contains_match: bool = False,
    timezone: str = "America/New_York",
) -> str:
    """Delete events matching a title within an optional time window. Returns JSON."""
    cal = _get_calendar_by_name(calendar_name)
    zone = pytz.timezone(timezone)

    def _parse_opt(dt_str: Optional[str], default_dt: datetime) -> datetime:
        if not dt_str:
            dt = default_dt
        else:
            if "T" in dt_str:
                dt = datetime.fromisoformat(dt_str)
            else:
                d = date.fromisoformat(dt_str)
                dt = datetime.combine(d, time(0, 0))
        if dt.tzinfo is None:
            dt = zone.localize(dt)
        return dt

    default_start = zone.localize(datetime.now()) - timedelta(days=365)
    default_end = zone.localize(datetime.now()) + timedelta(days=730)
    start = _parse_opt(starts_after_iso, default_start)
    end = _parse_opt(ends_before_iso, default_end)

    deleted = []
    for ev in cal.date_search(start, end):
        if len(deleted) >= max_delete:
            break
        try:
            ics = Calendar.from_ical(ev.data)
            for comp in ics.walk("VEVENT"):
                s = str(comp.get("summary", ""))
                lhs = s if case_sensitive else s.casefold()
                rhs = summary if case_sensitive else summary.casefold()
                match = (rhs in lhs) if contains_match else (lhs == rhs)
                if match:
                    uid = str(comp.get("uid", ""))
                    ev.delete()
                    deleted.append({
                        "summary": s,
                        "uid": uid,
                        # Cast CalDAV URL object to string
                        "href": _safe_str(getattr(ev, "url", "")),
                    })
                    break
        except Exception:
            continue

    # Robust JSON: default=_safe_str handles any other non-serializable objects
    return json.dumps({"deleted_count": len(deleted), "deleted": deleted}, ensure_ascii=False, default=_safe_str)



# --- 2) delete_multiple_calendar_events_tool replacement ---
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
import json
from datetime import datetime, timedelta, date, time
import pytz

@tool("delete_multiple_calendar_events")
def delete_multiple_calendar_events_tool(
    instruction_text: str,
    default_calendar_name: str = "Assignments",
    timezone: str = "America/New_York",
    max_delete_per_title: int = 4,
    contains_match: bool = False,
) -> str:
    """Delete multiple events. Parse titles and an optional date, then call the single-delete tool per title.

    Behavior:
    - Resolve calendar names against the current iCloud calendars (cached once).
    - If a date is present, delete only within that day window [00:00, +1d).
    - Calls `delete_calendar_event` once per title.
    Returns JSON: {"results":[{summary,calendar_name,starts_after_iso,ends_before_iso,deleted_count,deleted,error}]}
    """
    import json
    from langchain_core.prompts import ChatPromptTemplate

    class DeleteSpec(BaseModel):
        summary: str
        calendar_name: Optional[str] = None
        date_iso: Optional[str] = None

    class DeletePlan(BaseModel):
        events: List[DeleteSpec]

    # Plan extraction
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Return ONLY JSON. Top-level key 'events'. Each event: summary; optional calendar_name; optional date_iso (YYYY-MM-DD). "
         "Do not invent dates. If one date applies to many titles, repeat it."),
        ("user", "{text}")
    ])
    planner = prompt | llm.with_structured_output(DeletePlan)
    plan = planner.invoke({"text": instruction_text})

    # Cache calendars **once**
    try:
        calendars = json.loads(list_calendars_tool.invoke({}))
    except Exception:
        calendars = []

    def resolve_calendar_name(requested: Optional[str], fallback: str) -> str:
        if not calendars:
            return requested or fallback
        target = (requested or fallback or "").casefold()
        names = [c.get("displayName", "") for c in calendars]
        lower = [n.casefold() for n in names]
        if target in lower:
            return names[lower.index(target)]
        for i, n in enumerate(lower):
            if target and target in n:
                return names[i]
        return fallback

    def day_window(date_str: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        if not date_str:
            return None, None
        z = pytz.timezone(timezone)
        d = date.fromisoformat(date_str)
        start_dt = z.localize(datetime.combine(d, time(0, 0, 0)))
        end_dt = start_dt + timedelta(days=1)
        return start_dt.isoformat(), end_dt.isoformat()

    results = []
    for ev in plan.events:
        cal_name = resolve_calendar_name(ev.calendar_name, default_calendar_name)
        start_iso, end_iso = day_window(ev.date_iso)
        raw = delete_calendar_event_tool.invoke({
            "summary": ev.summary,
            "calendar_name": cal_name,
            "starts_after_iso": start_iso,
            "ends_before_iso": end_iso,
            "max_delete": max_delete_per_title,
            "case_sensitive": False,
            "contains_match": contains_match,
            "timezone": timezone,
        })
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            deleted_count = parsed.get("deleted_count")
            deleted = parsed.get("deleted")
            error = None
        except Exception as e:
            deleted_count, deleted, error = None, None, f"json_error: {e}"
        results.append({
            "summary": ev.summary,
            "calendar_name": cal_name,
            "starts_after_iso": start_iso,
            "ends_before_iso": end_iso,
            "deleted_count": deleted_count,
            "deleted": deleted,
            "error": error,
        })

    return json.dumps({"results": results})

calendar_tools = [
    list_calendars_tool,
    create_multiple_calendar_events_tool,
    delete_multiple_calendar_events_tool,
    create_calendar_event_tool,
    delete_calendar_event_tool,
]

calendar_agent_description = (
    "You are a Calendar agent. create/update Apple Calendar events. "
    "Use list_calendars FIRST to confirm available names. "
    "For multiple event creation/deletion, call create_multiple_calendar_events or delete_multiple_calendar_events EXACTLY ONCE with default_calendar_name='Assignments'. "
    "For single event creation/deletion, call create_calendar_event or delete_calendar_event. "
    "If only a date is provided, time defaults to 10:59–11:59 PM local time."
)

CALENDAR_SYSTEM = (
    "You are a calendar operations agent. You MUST use the provided tools. "
    "Always call list_calendars as the FIRST step to see valid calendar names. "
    "If the user requests TWO OR MORE event creations, call create_multiple_calendar_events EXACTLY ONCE "
    "If the user requests TWO OR MORE event deletions, call delete_multiple_calendar_events EXACTLY ONCE "
    "(same defaults). "
    "If ONE event: use create_calendar_event or delete_calendar_event. "
    "Do not invent dates or times. For date-only, pass date-only and omit end_iso. "
    "Do not give the user UIDs unless prompted to."
)

CALENDAR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CALENDAR_SYSTEM),
    ("placeholder", "{messages}")
])
calendar_agent = create_react_agent(llm, tools=calendar_tools, prompt=CALENDAR_PROMPT)


############################################
# Supervisor (router)

MAX_ITERATIONS = 7

worker_descriptions = {
    "calendar_agent": calendar_agent_description,
}

formatted_descriptions = "\n".join(
    f"- {name}: {desc}" for name, desc in worker_descriptions.items()
)

system_prompt = (
    "Decide between routing between the following workers or ending the conversation if an answer is provided.\n"
    + formatted_descriptions
)
options = ["FINISH"] + list(worker_descriptions.keys())
FINISH = {"next_node": "FINISH"}


def supervisor_agent(state):
    count = state.get("iteration_count", 0) + 1
    if count > MAX_ITERATIONS:
        return FINISH

    class nextNode(BaseModel):
        next_node: Literal[tuple(options)]

    preprocessor = RunnableLambda(
        lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
    )
    supervisor_chain = preprocessor | llm.with_structured_output(nextNode)
    next_node = supervisor_chain.invoke(state).next_node

    if state.get("next_node") == next_node:
        return FINISH
    return {"iteration_count": count, "next_node": next_node}




# Multi-agent graph
#######################################


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [
            {
                "role": "assistant",
                "content": result["messages"][-1].content,
                "name": name,
            }
        ]
    }


# def final_answer(state):
#     prompt = (
#         "Using only the content in the messages, respond to the previous user question using the answer given by the other assistant messages."
#     )
#     preprocessor = RunnableLambda(
#         lambda state: state["messages"] + [{"role": "user", "content": prompt}]
#     )
#     final_answer_chain = preprocessor | llm
#     return {"messages": [final_answer_chain.invoke(state)]}


class AgentState(ChatAgentState):
    next_node: str
    iteration_count: int


calendar_node = functools.partial(agent_node, agent=calendar_agent, name="calendar_agent")

workflow = StateGraph(AgentState)

workflow.add_node("calendar_agent", calendar_node)
workflow.add_node("supervisor", supervisor_agent)

workflow.set_entry_point("supervisor")
for worker in worker_descriptions.keys():
    workflow.add_edge(worker, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next_node"],
    {**{k: k for k in worker_descriptions.keys()}, "FINISH": END},
)
# workflow.add_edge("final_answer", END)
multi_agent = workflow.compile()


# from IPython.display import display, Image
# display(Image(multi_agent.get_graph().draw_mermaid_png()))


###################################
# Wrap in ChatAgent for serving



class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": [m.model_dump_compat(exclude_none=True) for m in messages]}

        messages_out = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages_out.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages_out)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": [m.model_dump_compat(exclude_none=True) for m in messages]}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data.get("messages", [])
                )


mlflow.langchain.autolog()
AGENT = LangGraphChatAgent(multi_agent)
mlflow.models.set_model(AGENT)
