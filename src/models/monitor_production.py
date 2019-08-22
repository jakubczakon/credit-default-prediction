import neptune

from src.models.utils import ProductionMonitor

RESOURCE_MONITOR_URL = 'https://play.grafana.org/d/000000012/grafana-play-home?orgId=1'
EXP_ID = 'CRED-81'

session = neptune.sessions.Session()
project = session.get_project('neptune-ml/credit-default-prediction')
exp = project.get_experiments(id=EXP_ID)[0]

exp.append_tag('production')
exp.set_property('production_resources', RESOURCE_MONITOR_URL)

monitor = ProductionMonitor()
while monitor.running:
    precision, recall = monitor.get_metrics()
    exp.send_metric('production_precision', precision)
    exp.send_metric('production_recall', recall)