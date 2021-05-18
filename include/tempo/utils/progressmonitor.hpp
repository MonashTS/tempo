#pragma once

#include <ostream>

namespace tempo {

  struct ProgressMonitor {
    // --- --- --- Init Fields
    size_t total{0};         // E.g. 390
    size_t tenth{0};         // E.g. 39
    size_t centh{0};         // E.g. 3
    size_t extra_centh{0};   // Extra 'centh' per tenth: with the above: 9


    // --- --- --- Constructor
    explicit ProgressMonitor(size_t total){
      this->total = total;
      tenth = total/10;
      centh = total/100;
      if(centh!=0){ extra_centh = (tenth-centh*10)/centh; }
    }

    // --- --- --- Print progress
    void print_progress(std::ostream& out, size_t nbdone){
      size_t nbtenth = 0;
      if(tenth!=0){nbtenth = nbdone/tenth; }
      size_t nbcenth = 0;
      if(centh != 0){ nbcenth = nbdone/centh; }
      size_t limit = (1+nbtenth)*10+(nbtenth*extra_centh);
      if(tenth!=0 && nbdone%tenth==0){
        out << nbtenth*10 << "% ";
      } else if (centh == 0 || nbdone%centh==0 && nbcenth < limit) {
        out << ".";
      }
      std::flush(out);
    }

  };

}