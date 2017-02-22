class SearchConfig:
    def __init__(self,x_start_stop=[None, None], y_start_stop=[None, None],
                        xy_window=(96, 96), xy_overlap=(0.5, 0.5)):
        self.x_start_stop=x_start_stop
        self.y_start_stop=y_start_stop
        self.xy_window=xy_window
        self.xy_overlap=xy_overlap
