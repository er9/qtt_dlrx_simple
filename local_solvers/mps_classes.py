from setup_.defaults import *

class MPS(qtn.MatrixProductState):

    _cur_orthog: int = None
    select_inds: dict[int, Sequence[int]] = None
    select_tens: dict[int, 'qtn.Tensor'] = None
    select_tens_inv: dict[int, 'qtn.Tensor'] = None

    @classmethod
    def from_quimb(cls, mps: 'qtn.MatrixProductState'):
        return MPS(mps)

    @property
    def cur_orthog(self):
        return self._cur_orthog

    def copy(self, virtual=False, deep=True):
        out = MPS(self, virtual=virtual, deep=deep)
        out._cur_orthog = self._cur_orthog
        if self.select_inds is not None:
            out.select_inds = {k:[*v] for k,v in self.select_inds.items()} # if deep else self.select_inds
        if self.select_tens is not None:
            out.select_tens = {k: v.copy() for k, v in self.select_tens.items()}
        if self.select_tens_inv is not None:
            out.select_tens_inv = {k: v.copy() for k, v in self.select_tens_inv.items()}
        return out

    def get_tens(self, i: int) -> Optional[qtn.Tensor]:
        try:
            return self[i]
        except IndexError:
            return None

    def bond(self, i:int, j:int):
        if (i == 0 and j == -1) or (i == -1 and j == 0):
            tens = self[0]
            bond_r = super().bond(0, 1)
            phys_i = self.site_ind(0)
            for ind in tens.inds:
                if ind != bond_r and ind != phys_i:
                    return ind
            return None
        elif (i == self.L and j == self.L-1) or (i == self.L-1 and j == self.L):
            tens = self[self.L-1]
            bond_l = super().bond(self.L-1, self.L-2)
            phys_i = self.site_ind(self.L-1)
            for ind in tens.inds:
                if ind != bond_l and ind != phys_i:
                    return ind
            return None
        else:
            return super().bond(i, j)

    def get_bra_tens(self, i: int, reindex_phys=False) -> Optional[qtn.Tensor]:
        tens = self.get_tens(i)
        if tens is None:
            return None
        bra_tens = tens.conj()  # self[i].conj()
        phys_ind = self.site_ind(i)
        bra_tens.modify(inds=[(ind if (ind == phys_ind and not reindex_phys) else ind + '_')
                              for ind in bra_tens.inds])
        return bra_tens

    def canonize(self, where, cur_orthog='calc', bra=None):
        # if self.cur_orthog is not None:
        #     cur_orthog = self.cur_orthog
        super().canonize(where, cur_orthog=cur_orthog, bra=bra)
        self._cur_orthog = where
        if bra is not None:
            bra._cur_orthog = where

    def get_select_inds(self, left_site: int, right_site: int):

        from local_solvers.helper_mixed import tensor_get_submat

        indL, indR = left_site, right_site
        direction = 1 if indL < indR else -1
        if indL == indR:
            return self

        if self.select_inds is None:
            self.select_inds = {}
        if self.select_tens is None:
            self.select_tens = {}
        if self.select_tens_inv is None:
            self.select_tens_inv = {}

        if direction > 0:
            prev_tens = self.select_tens[indL - 1].copy() if indL > 0 else None
        else:
            prev_tens = self.select_tens[indL + 1].copy() if indL < self.L - 1 else None

        for i in range(indL, indR, direction):
            tens_i = self[i]
            tens = tens_i.copy() if prev_tens is None else qtn.tensor_contract(prev_tens, tens_i)

            pbond = self.site_ind_id.format(i)
            lbond = self.bond(i, i - direction)
            if lbond is not None:
                lbond = lbond + '_x'   ## because of contraction with prev_tens
            rbond = self.bond(i, i + direction)

            inds_r, TC, TR_inv, TR = tensor_get_submat(tens, lbond, rbond, pbond)
            ## TC: rbond + '_'  //  rbond  (throw away)
            ## TU: rbond  //  rbond + '_x'
            ## TR: rbond + 'x' // rbond

            self.select_inds[i] = inds_r
            self.select_tens[i] = TR
            self.select_tens_inv[i] = TR_inv

            prev_tens = TR

        return self

    def check_select_inds(self):
        indL = self.check_left_select_inds()
        indR = self.check_right_select_inds()
        return indL, indR

    def check_left_select_inds(self) -> int:
        """ returns index of first tensor that is not left canonical
        """
        from local_solvers.helper_mixed import check_left_select_inds
        return check_left_select_inds(self, self.select_inds, self.select_tens, self.select_tens_inv)

    def check_right_select_inds(self) -> int:
        """ returns index of first tensor that is not left canonical
        """
        from local_solvers.helper_mixed import check_right_select_inds
        return check_right_select_inds(self, self.select_inds, self.select_tens, self.select_tens_inv)



MPO = qtn.MatrixProductOperator

# from local_solvers.helper_cross_2 import tensor_select_rows, tensor_xr
# from local_solvers.helper_mixed import tensor_get_submat