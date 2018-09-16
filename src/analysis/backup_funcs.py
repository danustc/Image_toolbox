    def spatial_gridding(self, ng = (2, 2, 2) ):
        '''
        classify cells into different small categories using spatial classification
        rev_coord: if true, the coordinates are arranged in z-y-x; otherwise, x-y-z.
        Warning: the order of coord and ng must be consistent, i.e., both z-y-x or both x-y-z.
        output: an array of raveled indices in the order of z-y-x.
        '''
        coord = self.coord
        H, edges = np.histogramdd(coord, bins = ng) # 3D histogram
        print("Number of bins:", H.shape)

        if self.rev:
            cz, cy, cx = coord[:,0], coord[:,1], coord[:,2]
            egz, egy, egx = edges
            bz, by, bx = ng
        else:
            mx, my, mz = coord.max(axis = 0)
            cx, cy, cz = coord[:,0], coord[:,1], coord[:,2]
            egx, egy, egz = edges
            bx, by, bz = ng

        egx[0] -=1.0e-06
        egy[0] -=1.0e-06
        egz[0] -=1.0e-06

        # find the indices of edges for each neuron
        ind_x = np.searchsorted(egx, cx) - 1
        ind_y = np.searchsorted(egy, cy) - 1
        ind_z = np.searchsorted(egz, cz) - 1
        print(ind_x.min(), ind_y.min(), ind_z.min())

        rav_label = np.ravel_multi_index((ind_z, ind_y, ind_x), (bz, by, bx))
        self.group_mark = rav_label #divide cells into groups
        return rav_label


