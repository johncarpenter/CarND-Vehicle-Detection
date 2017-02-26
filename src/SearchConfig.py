class SearchConfig:
    def __init__(self,x_start_stop=[None, None], y_start_stop=[None, None],
                        xy_window=(96, 96), xy_overlap=(0.5, 0.5)):
        '''
        Configuration class to hold the search window parameters
        '''
        self.x_start_stop=x_start_stop
        self.y_start_stop=y_start_stop
        self.xy_window=xy_window
        self.xy_overlap=xy_overlap
    def dump(self):
        '''
        Debug output
        '''
        print("X Range {}".format(self.x_start_stop))
        print("Y Range {}".format(self.y_start_stop))
        print("Window Size {}".format(self.xy_window))
        print("Overlap {}".format(self.xy_overlap))
