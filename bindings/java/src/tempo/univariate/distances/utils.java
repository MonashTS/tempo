package tempo.univariate.distances;

public class utils {
	static double EPSILON = 10e-12;
	
	static double dist(double a, double b) {
		double d = a-b;
		return d*d;
	}
	
	static double min(double a, double b, double c) {
		return Double.min(a, Double.min(b, c));
	}
	
}
