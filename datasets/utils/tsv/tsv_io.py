# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2021-08-13 14:26:21
# @Last Modified by:   Yihao Chen
# @Last Modified time: 2022-08-17 00:57:51
import time
import os
import os.path as op
from .io_common import generate_lineidx, FileProgressingbar


class TSVFile(object):
    def __init__(self, tsv_file, silence=True):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'

        self.label_file = op.splitext(tsv_file)[0] + '.label'
        self.label_lineidx = op.splitext(tsv_file)[0] + '.label.lineidx'

        if os.path.exists(self.label_file):
            self.split_label = True
        else:
            self.split_label = False

        self._fp = None
        self._lineidx = None

        self._label_fp = None
        self._label_lineidx = None

        self.pid = None
        self.silence = silence
        self._ensure_lineidx_loaded()

    def num_rows(self):
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        tsv_info = [s.strip() for s in self._fp.readline().split('\t')]    

        if self.split_label:
            label_pos = self._label_lineidx[idx]
            self._label_fp.seek(label_pos)
            label_info = [s.strip() for s in self._label_fp.readline().split('\t')]

            assert tsv_info[0] == label_info[0]
            tsv_info = [tsv_info[0], label_info[-1], tsv_info[-1]]

        return tsv_info

    def close(self):
        if self._fp is not None:
            self._fp.close()
            del self._fp
            del self._lineidx
            
            self._fp = None
            self._lineidx = None

    def _ensure_lineidx_loaded(self):
        if not op.isfile(self.lineidx) and not op.islink(self.lineidx):
            generate_lineidx(self.tsv_file, self.lineidx)

        if self._lineidx is None:
            with open(self.lineidx, 'r') as fp:
                lines = fp.readlines()
                self._lineidx = [int(i.strip().split()[0]) for i in lines]

            if self.split_label:
                with open(self.label_lineidx, 'r') as fp:
                    lines = fp.readlines()
                    self._label_lineidx = [int(i.strip().split()[0]) for i in lines]                


    def _ensure_tsv_opened(self):
        self._ensure_lineidx_loaded()
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

            if self.split_label:
                self._label_fp = open(self.label_file, 'r')

        if self.pid != os.getpid():
            print('re-open {} because the process id changed'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

            if self.split_label:
                self._label_fp = open(self.label_file, 'r')
