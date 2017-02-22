class Config:
    def __init__(self,color_space='HLS', spatial_size=(16, 16),
                        hist_bins=32, orient=9,
                        pix_per_cell=16, cell_per_block=2, hog_channel='ALL',
                        spatial_feat=True, hist_feat=True, hog_feat=True):
        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_bins= hist_bins
        self.orient= orient
        self.pix_per_cell= pix_per_cell
        self.cell_per_block= cell_per_block
        self.hog_channel= hog_channel
        self.spatial_feat= spatial_feat
        self.hist_feat= hist_feat
        self.hog_feat= hog_feat

    def dump(self):
        print("Color Space {}".format(self.color_space))
        print("Spatial Bin Size {}".format(self.spatial_size))
        print("Histogram Bins {}".format(self.hist_bins))
        print("Orientations {}".format(self.orient))
        print("Px per Cell {}".format(self.pix_per_cell))
        print("Cell per Block {}".format(self.cell_per_block))
        print("HOG Channel {}".format(self.hog_channel))
        print("Use Spatial Features {}".format(self.spatial_feat))
        print("Use Histogram Features {}".format(self.hist_feat))
        print("Use HOG Features {}".format(self.hog_feat))
