from agents.crl import CRLAgent
from agents.crl_infonce import CRLInfoNCEAgent
from agents.fmrl import FMRLAgent
from agents.fql import FQLAgent
from agents.gcfmrl import GCFMRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcsarsaql import GCSARSAQLAgent
from agents.gcmciql import GCMCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.rg_fmrl import RewardGuidedFMRLAgent
from agents.sac import SACAgent
from agents.td_fmrl import TDFMRLAgent
from agents.gctd_fmrl import GCTDFMRLAgent
from agents.td_infonce import TDInfoNCEAgent

agents = dict(
    crl=CRLAgent,
    crl_infonce=CRLInfoNCEAgent,
    fmrl=FMRLAgent,
    fql=FQLAgent,
    gcfmrl=GCFMRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcsarsaql=GCSARSAQLAgent,
    gcmciql=GCMCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    rg_fmrl=RewardGuidedFMRLAgent,
    sac=SACAgent,
    td_fmrl=TDFMRLAgent,
    gctd_fmrl=GCTDFMRLAgent,
    td_infonce=TDInfoNCEAgent,
)
