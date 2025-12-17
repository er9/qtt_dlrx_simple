from axis_map.map import AxisMap

""" mapping of grid points along 1 dimension to binary TN ordering
    MPS left to right = coarse grid to fine grid
"""

## binary TN class
class BinaryMap(AxisMap):

    @classmethod
    def get_position_inds(cls,L,q,idx):
        """ returns physical bond indices (0,1) of MPS that corresponds to vector position idx
        """
        if L == 1:
            return [idx]

        assert (q==2), 'atm method only implemented for q=2'

        if idx < 0:
            idx = q**L + idx

        if q==2:   str_b = bin(idx)[2:]
        else:
            if idx == 0:  str_b = [0]
            else:
                str_b = []
                while idx:
                    str_b.append(int(idx % q))
                    idx //= q
                str_b = str_b[::-1]

        if len(str_b) > L:
            print('ind not in binaryTN of len L',L)
            raise ValueError
        elif len(str_b) < L:
            str_b = '0'*(L-len(str_b)) + str_b

        ind_list = [int(s) for s in str_b]
        return ind_list


    @classmethod
    def get_inds_position(cls, q, inds_list):
        """ returns index in vector corresponding to physical bond indices specified in inds_list
        """
        bstring = ''
        for b in inds_list:   bstring += str(b)
        i = bstring
        idx = int(i,q)
        return idx


