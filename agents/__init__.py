from agents.brm import BRMAgent
from agents.brm_sfbc import BRMSfBCAgent
from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent

agents = dict(
    brm=BRMAgent,
    brm_sfbc=BRMSfBCAgent,
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
)
