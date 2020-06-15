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
        self.viofeed = []  # list to accumulate violations per update interval
        self.nonviofeed = []  # list to accumulate non violations per update interval
        self.sevidx = []  # list to accumulate severity index per update interval
        self.violocationsx = []  # list to accumulate violations coordinates per update interval
        self.violocationsy = []
        self.vio = 0  # final violations (sum(len(viofeed)))
        self.nonvio = 0  # final nonviolations (sum(len(nonviolations)))
        self.sevidxavg = 0.0  # average severity index per update interval

    def feed_new(self, new_feed: tuple):
        self.viofeed.append(new_feed[0])
        self.nonviofeed.append(new_feed[1])
        self.sevidx.append(new_feed[2])
        for pair in new_feed[0]:
            self.violocationsx.append(np.median([pair[0][0], pair[1][0]]))
            self.violocationsy.append(np.median([pair[0][1], pair[1][1]]))

    def get_feed(self):
        '''
        get update interval accumulations
        '''
        self.vio = sum([len(i) for i in self.viofeed])
        self.nonvio = sum([len(i) for i in self.nonviofeed])
        if len(self.sevidx) > 0:
            self.sevidxavg = np.mean(self.sevidx)
        else:
            self.sevidxavg = 0.0

        return self.vio, self.nonvio, self.sevidxavg, self.violocationsx, self.violocationsy

    def clear_feed(self):
        self.__init__()
