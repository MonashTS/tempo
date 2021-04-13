package tempo.univariate.distances;

public class utils {
	static double EPSILON = 10e-12;
	
	static double dist(double a, double b) {
		double d = a-b;
		return d*d;
	}
	
	static int min(int a, int b, int c) {
		return Integer.min(a, Integer.min(b, c));
	}
	
	static int max(int a, int b, int c) {
		return Integer.max(a, Integer.max(b, c));
	}
	
	static double min(double a, double b, double c) {
		return Double.min(a, Double.min(b, c));
	}
	
	static double max(double a, double b, double c) {
		return Double.max(a, Double.max(b, c));
	}
	
	
}
