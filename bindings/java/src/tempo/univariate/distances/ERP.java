package tempo.univariate.distances;

import static tempo.univariate.distances.utils.dist;
import static tempo.univariate.distances.utils.min;

import java.util.Arrays;

import static java.lang.Double.POSITIVE_INFINITY;
import static java.lang.Double.min;
import static java.lang.Integer.max;
import static java.lang.Integer.min;

public class ERP {
	
	static double distance(double[] lines, double[] cols, double gValue, int w, double cutoff) {
		
		// Ensure that lines are longer than columns
		if(lines.length < cols.length) {
			double [] swap = lines;
			lines = cols;
			cols = swap;
		}
		
        // Cap the windows and check that, given the constraint, an alignment is possible
        if (w > lines.length) { w = lines.length; }
        if (lines.length - cols.length > w) { return POSITIVE_INFINITY; }
		
		// --- --- --- Declarations
		int nblines = lines.length;
		int nbcols = cols.length;
				
		// Setup buffers - get an extra cell for border condition. Init to +INF.
		double[] buffers = new double[(1+nbcols)*2];
		Arrays.fill(buffers, POSITIVE_INFINITY);
		int c = 0+1; 		// Start of current line in buffer - account for the extra cell
		int p = nbcols+2;   // Start of previous line in buffer - account for two extra cells
		
		// Line & columns indices
		int i = 0;
		int j = 0;
		
		// Cost accumulator in a line, also used as the "left neighbor"
		double cost=0;
		
		// EAP variable: track where to start the next line, and the position of the previous pruning point.
		// Must be init to 0: index 0 is the next starting point and also the "previous pruning point"
		int next_start = 0;
		int prev_pp = 0;

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Create a new tighter upper bounds (most commonly used in the code).
        // First, take the "next float" after "cutoff" to deal with numerical instability.
        // Then, subtract the cost of the last alignment.
		double la = min(
                    dist(gValue, cols[nbcols - 1]),             // Previous
                    dist(lines[nblines - 1], cols[nbcols - 1]), // Diagonal
                    dist(lines[nblines - 1], gValue)            // Above
            );
		double ub = cutoff + utils.EPSILON - la;


        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Initialisation of the top border
        {   // Matrix Border - Top diagonal
            buffers[c - 1] = 0;
            // Matrix Border - First line
            int jStop =  min(i+w+1, nbcols);
            for (j = 0; buffers[c + j - 1] <= ub && j < jStop; ++j) {
                buffers[c + j] = buffers[c + j - 1] + dist(gValue, cols[j]);
            }
            // Pruning point set to first +INF value (or out of bound)
            prev_pp = j;
        }

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Part 1: Loop with computed left border.
        {   // The left border has a computed value while it's within the window and its value bv <= ub
            // No "front pruning" (next_start) and no early abandoning can occur while in this loop.
            int iStop =  min(i+w+1, nblines);
            for (; i < iStop; ++i) {
                // --- --- --- Variables init
                double li = lines[i];
                int jStart = 0;
                int jStop = min(i+w+1, nbcols);
                j = jStart;
                int curr_pp = jStart; // Next pruning point init at the start of the line
                // --- --- --- Stage 0: Initialise the left border
                {
                    // We haven't swap yet, so the 'top' cell is still indexed by 'c-1'.
                    cost = buffers[c - 1] + dist(li, gValue);
                    if (cost > ub) { break; }
                    else {
                    	int swap = c;
                    	c = p;
                    	p = swap;
                        buffers[c - 1] = cost;
                    }
                }
                // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
                // No stage 1 here.
                // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
                for (; j < prev_pp; ++j) {
                    cost = min(
                            cost + dist(gValue, cols[j]),               // Previous
                            buffers[p + j - 1] + dist(li, cols[j]),     // Diagonal
                            buffers[p + j] + dist(li, gValue)           // Above
                    );
                    buffers[c + j] = cost;
                    if (cost <= ub) { curr_pp = j + 1; }
                }
                // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
                if (j < jStop) { // Possible path in previous cells: left and diag.
                    cost = min(
                            cost + dist(gValue, cols[j]),               // Previous
                            buffers[p + j - 1] + dist(li, cols[j])      // Diagonal
                    );
                    buffers[c + j] = cost;
                    if (cost <= ub) { curr_pp = j + 1; }
                    ++j;
                }
                // --- --- --- Stage 4: After the previous pruning point: only prev.
                // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
                for (; j == curr_pp && j < jStop; ++j) {
                    cost = cost + dist(gValue, cols[j]);  // Previous
                    buffers[c + j] = cost;
                    if (cost <= ub) { ++curr_pp; }
                }
                // --- --- ---
                prev_pp = curr_pp;
            }
        }

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Part 2: Loop with +INF left border
        {
            for (; i < nblines; ++i) {
                // --- --- --- Swap and variables init
            	int swap = c;
            	c = p;
            	p = swap;
                double li = lines[i];
                int jStart = max(i-w, next_start);
                int jStop = min(i+w+1, nbcols); 
                j = jStart;
                next_start = jStart;
                int curr_pp = jStart; // Next pruning point init at the start of the line
                // --- --- --- Stage 0: Initialise the left border
                {
                    cost = POSITIVE_INFINITY;
                    buffers[c + jStart - 1] = cost;
                }
                // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
                for (; j == next_start && j < prev_pp; ++j) {
                    cost = min(
                            buffers[p + j - 1] + dist(li, cols[j]),     // Diagonal
                            buffers[p + j] + dist(li, gValue)           // Above
                    );
                    buffers[c + j] = cost;
                    if (cost <= ub) { curr_pp = j + 1; } else { ++next_start; }
                }
                // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
                for (; j < prev_pp; ++j) {
                    cost = min(
                            cost + dist(gValue, cols[j]),               // Previous
                            buffers[p + j - 1] + dist(li, cols[j]),     // Diagonal
                            buffers[p + j] + dist(li, gValue)           // Above
                    );
                    buffers[c + j] = cost;
                    if (cost <= ub) { curr_pp = j + 1; }
                }
                // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
                if (j < jStop) { // If so, two cases.
                    if (j == next_start) { // Case 1: Advancing next start: only diag.
                        cost = buffers[p + j - 1] + dist(li, cols[j]);     // Diagonal
                        buffers[c + j] = cost;
                        if (cost <= ub) { curr_pp = j + 1; }
                        else {
                            // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
                            if (i == nblines - 1 && j == nbcols - 1 && cost <= cutoff) { return cost; }
                            else { return POSITIVE_INFINITY; }
                        }
                    } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
                        cost = min(
                                cost + dist(gValue, cols[j]),               // Previous
                                buffers[p + j - 1] + dist(li, cols[j])      // Diagonal
                        );
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
                for (; j == curr_pp && j < jStop; ++j) {
                    cost = cost + dist(gValue, cols[j]);
                    buffers[c + j] = cost;
                    if (cost <= ub) { ++curr_pp; }
                }
                // --- --- ---
                prev_pp = curr_pp;
            }
        }

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Finalisation
        // Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
        if (j == nbcols && cost <= cutoff) { return cost; }
        else { return POSITIVE_INFINITY; }
    }

}
