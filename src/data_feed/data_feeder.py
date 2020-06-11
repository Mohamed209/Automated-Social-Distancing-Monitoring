import datetime
import numpy as np


class DataFeed:
    '''
    acts as base class for data sources pipes
    '''

    def __init__(self):
        pass

    def feed_new(self):
        pass

    def get_feed(self):
        pass

    def clear_feed(self):
        pass


class ViolationsFeed(DataFeed):
    '''
    class mainly responsible for feeding violations data for frontend
    '''

    def __init__(self):
        super().__init__()
        self.viofeed = []
        self.nonviofeed = []
        self.sevidx = []
        self.vio = 0
        self.nonvio = 0
        self.sevidxavg = 0.0

    def feed_new(self, new_feed: tuple):
        self.viofeed.append(new_feed[0])
        self.nonviofeed.append(new_feed[1])
        self.sevidx.append(new_feed[2])

    def get_feed(self):
        self.vio = sum([len(i) for i in self.viofeed])
        self.nonvio = sum([len(i) for i in self.nonviofeed])
        print("sev list", self.sevidx)
        if len(self.sevidx) > 0:
            self.sevidxavg = np.mean(self.sevidx)
        else:
            self.sevidxavg = 0.0
        return self.vio, self.nonvio, self.sevidxavg

    def clear_feed(self):
        self.__init__()
