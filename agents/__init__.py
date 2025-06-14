from agents.fdrl import FDRLAgent
from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent

agents = dict(
    fdrl=FDRLAgent,
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
)
