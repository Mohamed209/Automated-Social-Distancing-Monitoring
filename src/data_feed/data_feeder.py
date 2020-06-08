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
        self.vio = 0
        self.nonvio = 0

    def feed_new(self, new_feed: tuple):
        self.viofeed.append(new_feed)

    def get_feed(self):
        self.vio = [len(i[0]) for i in self.viofeed]
        self.nonvio = [len(i[1]) for i in self.viofeed]
        return self.vio, self.nonvio

    def clear_feed(self):
        self.__init__()
