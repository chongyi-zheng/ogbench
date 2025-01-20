from policy_evaluation.crl import CRLEstimator
# from agents.crl_infonce import CRLInfoNCEAgent
from policy_evaluation.fmrl import FMRLEstimator
# from agents.gcbc import GCBCAgent
# from agents.gciql import GCIQLAgent
# from agents.gcsarsaql import GCSARSAQLAgent
# from agents.gcmciql import GCMCIQLAgent
# from agents.gcivl import GCIVLAgent
# from agents.hiql import HIQLAgent
# from agents.qrl import QRLAgent
# from agents.sac import SACAgent
from policy_evaluation.td_infonce import TDInfoNCEEstimator

estimators = dict(
    crl=CRLEstimator,
    # crl_infonce=CRLInfoNCEAgent,
    fmrl=FMRLEstimator,
    # gcbc=GCBCAgent,
    # gciql=GCIQLAgent,
    # gcsarsaql=GCSARSAQLAgent,
    # gcmciql=GCMCIQLAgent,
    # gcivl=GCIVLAgent,
    # hiql=HIQLAgent,
    # qrl=QRLAgent,
    # sac=SACAgent,
    td_infonce=TDInfoNCEEstimator,
)
