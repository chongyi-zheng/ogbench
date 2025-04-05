from agents.crl import CRLAgent
from agents.crl_infonce import CRLInfoNCEAgent
from agents.fac import FACAgent
from agents.fmrl import FMRLAgent
from agents.fql import FQLAgent
from agents.gcbc import GCBCAgent
from agents.gcfac import GCFlowActorCriticAgent
from agents.gcfmrl import GCFMRLAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.gcmciql import GCMCIQLAgent
from agents.gcsarsaql import GCSARSAQLAgent
from agents.gctd_fmrl import GCTDFMRLAgent
from agents.hiql import HIQLAgent
from agents.ifac import IFACAgent
from agents.iql import IQLAgent
from agents.mcfac import MCFACAgent
from agents.qrl import QRLAgent
from agents.rebrac import ReBRACAgent
from agents.rg_fmrl import RewardGuidedFMRLAgent
from agents.sac import SACAgent
from agents.sarsa_ifac import SARSAIFACAgent
from agents.sarsa_ifac_q import SARSAIFACQAgent
from agents.sarsa_ifql import SARSAIFQLAgent
from agents.td_fmrl import TDFMRLAgent
from agents.td_infonce import TDInfoNCEAgent

agents = dict(
    crl=CRLAgent,
    crl_infonce=CRLInfoNCEAgent,
    fac=FACAgent,
    fmrl=FMRLAgent,
    fql=FQLAgent,
    gcfac=GCFlowActorCriticAgent,
    gcfmrl=GCFMRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcsarsaql=GCSARSAQLAgent,
    gcmciql=GCMCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    mcfac=MCFACAgent,
    ifac=IFACAgent,
    iql=IQLAgent,
    qrl=QRLAgent,
    rebrac=ReBRACAgent,
    rg_fmrl=RewardGuidedFMRLAgent,
    sac=SACAgent,
    sarsa_ifac=SARSAIFACAgent,
    sarsa_ifac_q=SARSAIFACQAgent,
    sarsa_ifql=SARSAIFQLAgent,
    td_fmrl=TDFMRLAgent,
    gctd_fmrl=GCTDFMRLAgent,
    td_infonce=TDInfoNCEAgent,
)
