import time
import numpy as np


class ProductionMonitor:    
    @property
    def running(self):
        time.sleep(5)
        return True
    
    def get_metrics(self):
        print('getting metrics')
        precision = 0.9 + 0.1 * np.random.random()
        recall = 0.8 + 0.2 * np.random.random()
        return precision, recall