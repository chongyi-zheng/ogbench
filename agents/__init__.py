from agents.fdrl import FDRLAgent
from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.mc_fdrl import MCFDRLAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent

agents = dict(
    fdrl=FDRLAgent,
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    mc_fdrl=MCFDRLAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
)
