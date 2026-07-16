from pathlib import Path
from ticket_agents.web_entry import make_web_agent
root_agent = make_web_agent("E09", Path(__file__).with_name("prompt.md"))
