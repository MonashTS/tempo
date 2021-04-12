package tempo.univariate.distances;

import static tempo.univariate.distances.utils.dist;
import static tempo.univariate.distances.utils.min;
import static java.lang.Double.POSITIVE_INFINITY;
import static java.lang.Double.min;



public class DTW {
	
	static double distance(double[] lines, double[] cols, double cutoff) {	
		// Ensure that lines are longer than columns
		if(lines.length < cols.length) {
			double [] swap = lines;
			lines = cols;
			cols = swap;
		}
		
		// --- --- --- Declarations
		int nblines = lines.length;
		int nbcols = cols.length;
		
		// Setup buffers - no extra initialization required - border condition managed in the code.
		double[] buffers = new double[2*nbcols];
		int c = 0;
		int p = nbcols;
		
		// Line & columns indices
		int i = 0;
		int j = 0;
		
		// Cost accumulator in a line, also used as the "left neighbor"
		double cost=0;
		
		// EAP variable: track where to start the next line, and the position of the previous pruning point.
		// Must be init to 0: index 0 is the next starting point and also the "previous pruning point"
		int next_start = 0;
		int prev_pp = 0;

		// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---		
		// Upper bound: tightened using the last alignment (requires special handling in the code below)
		// Add EPSILON helps dealing with numerical instability
		double ub=cutoff + utils.EPSILON - dist(lines[nblines - 1], cols[nbcols - 1]);
		
		// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
		// Initialization of the first line
        {
            double l0 = lines[0];
            // Fist cell is a special case.
            // Check against the original upper bound dealing with the case where we have both series of length 1.
            cost = dist(l0, cols[0]);
            if (cost > cutoff) { return POSITIVE_INFINITY; }
            buffers[c + 0] = cost;
            // All other cells. Checking against "ub" is OK as the only case where the last cell of this line is the
            // last alignment is taken are just above (1==nblines==nbcols, and we have nblines >= nbcols).
            int curr_pp = 1;
            for (j = 1; j == curr_pp && j < nbcols; ++j) {
                cost = cost + dist(l0, cols[j]);
                buffers[c + j] = cost;
                if (cost <= ub) { ++curr_pp; }
            }
            ++i;
            prev_pp = curr_pp;
        }
        
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Main loop
        for (; i < nblines; ++i) {
            // --- --- --- Swap and variables init
        	int swap = c;
        	c=p;
        	p=swap;

            double li = lines[i];
            int curr_pp = next_start; // Next pruning point init at the start of the line
            j = next_start;
            // --- --- --- Stage 0: Special case for the first column. Can only look up (border on the left)
            {
                cost = buffers[p + j] + dist(li, cols[j]);
                buffers[c + j] = cost;
                if (cost <= ub) { curr_pp = j + 1; } else { ++next_start; }
                ++j;
            }
            // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
            for (; j == next_start && j < prev_pp; ++j) {
                cost = min(buffers[p + j - 1], buffers[p + j]) + dist(li, cols[j]);
                buffers[c + j] = cost;
                if (cost <= ub) { curr_pp = j + 1; } else { ++next_start; }
            }
            // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
            for (; j < prev_pp; ++j) {
                cost = min(cost, buffers[p + j - 1], buffers[p + j]) + dist(li, cols[j]);
                buffers[c + j] = cost;
                if (cost <= ub) { curr_pp = j + 1; }
            }
            // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
            if (j < nbcols) { // If so, two cases.
                if (j == next_start) { // Case 1: Advancing next start: only diag.
                    cost = buffers[p + j - 1] + dist(li, cols[j]);
                    buffers[c + j] = cost;
                    if (cost <= ub) { curr_pp = j + 1; }
                    else {
                        // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
                        if (i == nblines - 1 && j == nbcols - 1 && cost <= cutoff) { return cost; }
                        else { return POSITIVE_INFINITY; }
                    }
                } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
                    cost = min(cost, buffers[p + j - 1]) + dist(li, cols[j]);
                    buffers[c + j] = cost;
                    if (cost <= ub) { curr_pp = j + 1; }
                }
                ++j;
            } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
                if (j == next_start) {
                    // But only if we are above the original UB
                    // Else set the next starting point to the last valid column
                    if (cost > cutoff) { return POSITIVE_INFINITY; }
                    else { next_start = nbcols - 1; }
                }
            }
            // --- --- --- Stage 4: After the previous pruning point: only prev.
            // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
            for (; j == curr_pp && j < nbcols; ++j) {
                cost = cost + dist(li, cols[j]);
                buffers[c + j] = cost;
                if (cost <= ub) { ++curr_pp; }
            }
            // --- --- ---
            prev_pp = curr_pp;
        } // End of main loop for(;i<nblines;++i)

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Finalization
        // Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
        if (j == nbcols && cost <= cutoff) { return cost; }
        else { return POSITIVE_INFINITY; }
	}

}
