from agents.crl import CRLAgent
from agents.crl_infonce import CRLInfoNCEAgent
from agents.dino_rebrac import DINOReBRACAgent
from agents.fac import FACAgent
from agents.fb_repr import ForwardBackwardRepresentationAgent
from agents.fb_repr_fom import ForwardBackwardRepresentationFOMAgent
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
from agents.hilp import HILPAgent
from agents.hilp_fom import HILPFOMAgent
from agents.hiql import HIQLAgent
from agents.ifac import IFACAgent
from agents.iql import IQLAgent
from agents.mcfac import MCFACAgent
from agents.mbpo_rebrac import MBPOReBRACAgent
from agents.qrl import QRLAgent
from agents.rebrac import ReBRACAgent
from agents.rg_fmrl import RewardGuidedFMRLAgent
from agents.sac import SACAgent
from agents.sarsa_ifac import SARSAIFACAgent
from agents.sarsa_ifac_q import SARSAIFACQAgent
from agents.sarsa_ifql import SARSAIFQLAgent
from agents.sarsa_ifql_gpi import SARSAIFQLGPIAgent
from agents.sarsa_ifql_vfm_gpi import SARSAIFQLVFMGPIAgent
from agents.sarsa_ifql_vib_gpi import SARSAIFQLVIBGPIAgent
from agents.td_fmrl import TDFMRLAgent
from agents.td_infonce import TDInfoNCEAgent

agents = dict(
    crl=CRLAgent,
    crl_infonce=CRLInfoNCEAgent,
    dino_rebrac=DINOReBRACAgent,
    fac=FACAgent,
    fmrl=FMRLAgent,
    fql=FQLAgent,
    fb_repr=ForwardBackwardRepresentationAgent,
    fb_repr_fom=ForwardBackwardRepresentationFOMAgent,
    gcfac=GCFlowActorCriticAgent,
    gcfmrl=GCFMRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcsarsaql=GCSARSAQLAgent,
    gcmciql=GCMCIQLAgent,
    gcivl=GCIVLAgent,
    hilp=HILPAgent,
    hilp_fom=HILPFOMAgent,
    hiql=HIQLAgent,
    mcfac=MCFACAgent,
    mbpo_rebrac=MBPOReBRACAgent,
    ifac=IFACAgent,
    iql=IQLAgent,
    qrl=QRLAgent,
    rebrac=ReBRACAgent,
    rg_fmrl=RewardGuidedFMRLAgent,
    sac=SACAgent,
    sarsa_ifac=SARSAIFACAgent,
    sarsa_ifac_q=SARSAIFACQAgent,
    sarsa_ifql=SARSAIFQLAgent,
    sarsa_ifql_vib_gpi=SARSAIFQLVIBGPIAgent,
    sarsa_ifql_vfm_gpi=SARSAIFQLVFMGPIAgent,
    sarsa_ifql_gpi=SARSAIFQLGPIAgent,
    td_fmrl=TDFMRLAgent,
    gctd_fmrl=GCTDFMRLAgent,
    td_infonce=TDInfoNCEAgent,
)
