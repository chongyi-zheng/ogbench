from agents.crl import CRLAgent
from agents.crl_infonce import CRLInfoNCEAgent
from agents.fmrl import FMRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcsarsaql import GCSARSAQLAgent
from agents.gcmciql import GCMCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.td_infonce import TDInfoNCEAgent

agents = dict(
    crl=CRLAgent,
    crl_infonce=CRLInfoNCEAgent,
    fmrl=FMRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcsarsaql=GCSARSAQLAgent,
    gcmciql=GCMCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    td_infonce=TDInfoNCEAgent,
)
