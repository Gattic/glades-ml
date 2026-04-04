// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "pca-test.h"
#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/image.h"
#include "../../../Backend/Machine Learning/GMath/pca.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "Backend/Plotter/Plotter.h"

// === This is the primary unit testing function:
// void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)

void createPCAImage(shmea::GString newImageName, const std::vector<std::vector<double> >& compute_data, const glades::PCA& pca)
{
    std::string imagePath = "datasets";
    std::string nameStr = newImageName.c_str(); // Convert GString to std::string

    const std::vector<std::vector<double> >& tdata = pca.getTransformedData();
    const std::vector<std::vector<double> > eigvecs = pca.getEigenvectors();
    const std::vector<std::vector<double> >& rdata = pca.getReconstructedData();

    // Create a plotter with default dimensions and supersampling factor of 4
    shmea::Plotter plotter(1800, 1000, 1);

    // Convert 2D data points to Plotter's Point format for original data
    std::vector<shmea::Point> original_points;
    for (size_t i = 0; i < compute_data.size(); ++i) {
        shmea::Point p;
        if (compute_data[i].size() >= 2) {
            p.x = compute_data[i][0];
            p.y = compute_data[i][1];
            original_points.push_back(p);
        }
    }

    // Save the original data plot
    plotter.chart()
        .title(nameStr + ": Original Data", 36)
        .grid(true)
        .axes(true)
        .cornerRadius(15)
        .logo("logo.png")
        .axisLabels("Feature 1", "Feature 2", 28)
        .originAxes(true)  // Enable four quadrant origin axes
        .autoMargins(shmea::CHART_SCATTER)
        .addSeries("Original Data", original_points, shmea::RGBA(0xFF, 0x00, 0x00, 0xFF), shmea::SERIES_SCATTER, 2, 8)
        .saveAs(nameStr + "_pca_original.png", imagePath);

    printf("[PCA] Original data image saved\n");

    // Create a second plotter for transformed data
    shmea::Plotter plotter2(1800, 1000, 1);

    // Convert 2D data points to Plotter's Point format for transformed data
    std::vector<shmea::Point> transformed_points;
    for (size_t i = 0; i < tdata.size(); ++i) {
        shmea::Point p;
        if (tdata[i].size() >= 2) {
            p.x = tdata[i][0];
            p.y = tdata[i][1];
            transformed_points.push_back(p);
        }
    }

    // Use the eigenvectors to create arrows for the diagram
    std::vector<shmea::Arrow> diagramArrows;
    for(unsigned int i = 0; i < eigvecs.size(); ++i)
    {
	shmea::Arrow arrow;
	arrow.start.x = 0.0f;
	arrow.start.y = 0.0f;
	arrow.end.x = eigvecs[i][0];
	arrow.end.y = eigvecs[i][1];
	arrow.color = shmea::RGBA(0x00, 0x00, 0xFF, 0xFF); // Blue color for arrows
	diagramArrows.push_back(arrow);
    }

    // Save the transformed data plot
    plotter2.chart()
        .title(nameStr + ": Transformed Data", 36)
        .grid(true)
        .axes(true)
        .cornerRadius(15)
        .logo("logo.png")
        .axisLabels("Principal Component 1", "Principal Component 2", 28)
        .originAxes(true)  // Enable four quadrant origin axes
        .autoMargins(shmea::CHART_SCATTER)
        .addSeries("Transformed Data", transformed_points, shmea::RGBA(0x00, 0xFF, 0x00, 0xFF), shmea::SERIES_SCATTER, 2, 8)
        .addArrows(diagramArrows)
        .saveAs(nameStr + "_pca_transformed.png", imagePath);

    printf("[PCA] Transformed data image saved\n");

    // Create a third plotter for reconstructed data
    shmea::Plotter plotter3(1800, 1000, 1);

    // Convert 2D data points to Plotter's Point format for reconstructed data
    std::vector<shmea::Point> reconstructed_points;
    for (size_t i = 0; i < rdata.size(); ++i) {
        shmea::Point p;
        if (rdata[i].size() >= 2) {
            p.x = rdata[i][0];
            p.y = rdata[i][1];
            reconstructed_points.push_back(p);
        }
    }

    // Save the reconstructed data plot
    plotter3.chart()
        .title(nameStr + ": Reconstructed Data", 36)
        .grid(true)
        .axes(true)
        .cornerRadius(15)
        .logo("logo.png")
        .axisLabels("Feature 1", "Feature 2", 28)
        .originAxes(true)  // Enable four quadrant origin axes
        .autoMargins(shmea::CHART_SCATTER)
        .addSeries("Reconstructed Data", reconstructed_points, shmea::RGBA(0xFF, 0x00, 0xFF, 0xFF), shmea::SERIES_SCATTER, 2, 8)
        .saveAs(nameStr + "_pca_reconstructed.png", imagePath);

    printf("[PCA] Reconstructed data image saved\n");
}

static void assertPCAProperties(const char* name, const glades::PCA& pca,
    const std::vector<std::vector<double> >& original_data, size_t expected_features)
{
    // Convergence
    ASSERT("PCA converged", pca.converged());

    const std::vector<double>& evals = pca.getEigenvalues();
    const std::vector<double>& ve = pca.getVarianceExplained();
    const std::vector<std::vector<double> > evecs = pca.getEigenvectors();
    const std::vector<std::vector<double> >& rdata = pca.getReconstructedData();

    ASSERT("PCA eigenvalues count", evals.size() == expected_features);
    ASSERT("PCA eigenvectors count", evecs.size() == expected_features);

    // Eigenvalues sorted descending
    for (size_t i = 0; i + 1 < evals.size(); ++i)
        ASSERT("PCA eigenvalues descending", evals[i] >= evals[i + 1] - 1e-10);

    // Variance explained sums to ~1.0
    double ve_sum = 0.0;
    for (size_t i = 0; i < ve.size(); ++i)
        ve_sum += ve[i];
    ASSERT("PCA variance_explained sums to ~1.0", std::fabs(ve_sum - 1.0) < 1e-6);

    // Eigenvector orthogonality (all pairs)
    for (size_t i = 0; i < evecs.size(); ++i)
    {
        // Unit length
        double mag = 0.0;
        for (size_t k = 0; k < evecs[i].size(); ++k)
            mag += evecs[i][k] * evecs[i][k];
        ASSERT("PCA eigenvector unit length", std::fabs(mag - 1.0) < 1e-6);

        for (size_t j = i + 1; j < evecs.size(); ++j)
        {
            double d = 0.0;
            for (size_t k = 0; k < evecs[i].size(); ++k)
                d += evecs[i][k] * evecs[j][k];
            ASSERT("PCA eigenvectors orthogonal", std::fabs(d) < 1e-6);
        }
    }

    // Lossless reconstruction when k=num_features
    if (pca.getNumComponents() == expected_features)
    {
        ASSERT("PCA reconstructed data size", rdata.size() == original_data.size());
        double max_err = 0.0;
        for (size_t i = 0; i < original_data.size(); ++i)
        {
            for (size_t j = 0; j < original_data[i].size(); ++j)
            {
                double err = std::fabs(rdata[i][j] - original_data[i][j]);
                if (err > max_err)
                    max_err = err;
            }
        }
        ASSERT("PCA lossless reconstruction (k=all)", max_err < 1e-6);
    }

    // Transform/InverseTransform round-trip
    std::vector<std::vector<double> > projected = pca.transform(original_data);
    std::vector<std::vector<double> > roundtrip = pca.inverseTransform(projected);
    if (pca.getNumComponents() == expected_features)
    {
        double max_rt_err = 0.0;
        for (size_t i = 0; i < original_data.size(); ++i)
        {
            for (size_t j = 0; j < original_data[i].size(); ++j)
            {
                double err = std::fabs(roundtrip[i][j] - original_data[i][j]);
                if (err > max_rt_err)
                    max_rt_err = err;
            }
        }
        ASSERT("PCA transform/inverseTransform round-trip", max_rt_err < 1e-6);
    }

    // getOriginalFeatureIndex out-of-range sentinel
    size_t sentinel = pca.getOriginalFeatureIndex(9999);
    ASSERT("PCA out-of-range getOriginalFeatureIndex sentinel", sentinel == pca.getMean().size());
}

void PCAUnitTest(bool saveImages)
{
    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("PCA Test y=x\n");
    printf("-----------------------------------\n");
    // Generate example data
    std::vector<std::vector<double> > example_data;
    int graphSize = 200; // pos and neg
    for (int i = -graphSize; i < graphSize; ++i)
    {
	double x = static_cast<double>(i) / graphSize * 10.0;
	double y = x; // easy visual example for testing

	std::vector<double> point;
	point.push_back(x);
	point.push_back(y);
	example_data.push_back(point);
    }

    glades::PCA pca1;
    ASSERT("PCA y=x compute succeeds", pca1.compute(example_data));

    const std::vector<double>& ve1 = pca1.getVarianceExplained();
    for(unsigned int i = 0; i < ve1.size(); ++i)
	printf("Principal Component %u: %.4f%%\n", i, ve1[i]*100.0);

    assertPCAProperties("y=x", pca1, example_data, 2);

    if (saveImages)
	createPCAImage("yequalsx", example_data, pca1);

    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("PCA Test x*x\n");
    printf("-----------------------------------\n");

    // Generate example data
    example_data.clear();
    for (int i = -graphSize; i < graphSize; ++i)
    {
	double x = static_cast<double>(i) / graphSize * 10.0;
	double y = x*x;

	std::vector<double> point;
	point.push_back(x);
	point.push_back(y);
	example_data.push_back(point);
    }

    glades::PCA pca2;
    ASSERT("PCA x*x compute succeeds", pca2.compute(example_data));

    const std::vector<double>& ve2 = pca2.getVarianceExplained();
    for(unsigned int i = 0; i < ve2.size(); ++i)
	printf("Principal Component %u: %.4f%%\n", i, ve2[i]*100.0);

    assertPCAProperties("x*x", pca2, example_data, 2);

    if (saveImages)
	createPCAImage("xsquared", example_data, pca2);

    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("PCA Test Trig\n");
    printf("-----------------------------------\n");

    // Generate example data
    example_data.clear();
    for (int i = -graphSize; i < graphSize; ++i)
    {
	double x = static_cast<double>(i) / graphSize * 10.0;
	double y = 0.5 * x + 0.5 * std::sin(3.0 * x) + 0.5 * std::cos(2.0 * x) + 0.5 * std::sin(5.0 * x) + 0.5 * std::cos(7.0 * x);

	std::vector<double> point;
	point.push_back(x);
	point.push_back(y);
	example_data.push_back(point);
    }

    glades::PCA pca3;
    ASSERT("PCA trig compute succeeds", pca3.compute(example_data));

    const std::vector<double>& ve3 = pca3.getVarianceExplained();
    for(unsigned int i = 0; i < ve3.size(); ++i)
	printf("Principal Component %u: %.4f%%\n", i, ve3[i]*100.0);

    assertPCAProperties("trig", pca3, example_data, 2);

    if (saveImages)
	createPCAImage("trig", example_data, pca3);

    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("PCA Test Iris\n");
    printf("-----------------------------------\n");

    shmea::GString path = "datasets/iris.data";
    glades::DataInput* di = new glades::NumberInput();
    di->import(path);

    std::vector<std::vector<double> > compute_data;
    unsigned int trainSize = di->getTrainSize();
    for(unsigned int i = 0; i < trainSize; ++i)
    {
	shmea::GVector<float> cRow = di->getTrainRow(i);
	std::vector<double> dataVec;
	for(unsigned int j = 0; j < cRow.size(); ++j)
	{
	    dataVec.push_back(cRow[j]);
	}

	compute_data.push_back(dataVec);
    }

    glades::PCA pca4;
    ASSERT("PCA Iris compute succeeds", pca4.compute(compute_data));

    const std::vector<double>& ve4 = pca4.getVarianceExplained();
    for(unsigned int i = 0; i < ve4.size(); ++i)
	printf("Principal Component %u: %.4f%%\n", i, ve4[i]*100.0);

    assertPCAProperties("Iris", pca4, compute_data, compute_data.empty() ? 0 : compute_data[0].size());

    // Save a PNG representation of the PCA
    if (saveImages)
	createPCAImage("iris", compute_data, pca4);

    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("PCA Test High-Dimensional Synthetic\n");
    printf("-----------------------------------\n");

    // Generate synthetic high-dimensional data (200 samples x 50 features)
    // Uses a deterministic LCG to produce correlated features
    unsigned int hd_samples = 200;
    unsigned int hd_features = 50;
    std::vector<std::vector<double> > hd_data(hd_samples, std::vector<double>(hd_features, 0.0));

    // Deterministic pseudo-random number generator (LCG)
    unsigned long lcg_state = 42;
    for (unsigned int i = 0; i < hd_samples; ++i)
    {
	// Generate base values with correlation
	double base1 = static_cast<double>(i) / hd_samples * 10.0 - 5.0;
	double base2 = base1 * 0.5 + 1.0;
	for (unsigned int j = 0; j < hd_features; ++j)
	{
	    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
	    double noise = (static_cast<double>(lcg_state >> 33) / 2147483648.0 - 0.5) * 0.1;
	    // Mix correlated and noise components
	    if (j < 5)
		hd_data[i][j] = base1 * (1.0 + j * 0.2) + noise;
	    else if (j < 10)
		hd_data[i][j] = base2 * (1.0 + (j - 5) * 0.15) + noise;
	    else
		hd_data[i][j] = noise * (1.0 + j * 0.01);
	}
    }

    printf("[PCA] Computing PCA on %u samples with %u features...\n", hd_samples, hd_features);
    glades::PCA pca5;
    ASSERT("PCA HighDim compute succeeds", pca5.compute(hd_data));

    const std::vector<double>& ve5 = pca5.getVarianceExplained();
    unsigned int numToShow = ve5.size() < 10 ? ve5.size() : 10;
    printf("Top %u principal components:\n", numToShow);
    for(unsigned int i = 0; i < numToShow; ++i)
	printf("Principal Component %u: %.4f%%\n", i, ve5[i]*100.0);

    assertPCAProperties("HighDim", pca5, hd_data, hd_features);

    // Also test with reduced components (k < num_features)
    glades::PCA pca5_reduced;
    ASSERT("PCA HighDim reduced compute succeeds", pca5_reduced.compute(hd_data, 10));
    ASSERT("PCA HighDim reduced: converged", pca5_reduced.converged());
    ASSERT("PCA HighDim reduced: 10 active components", pca5_reduced.getNumComponents() == 10);
    const std::vector<std::vector<double> >& tdata_r = pca5_reduced.getTransformedData();
    if (!tdata_r.empty())
	ASSERT("PCA HighDim reduced: projected to 10 dims", tdata_r[0].size() == 10);

    delete di;

    // ============================================================
    // Edge-case tests
    // ============================================================

    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("PCA Edge Case: Single Feature\n");
    printf("-----------------------------------\n");

    {
	std::vector<std::vector<double> > single_feat;
	for (int i = 0; i < 100; ++i)
	{
	    std::vector<double> row;
	    row.push_back(static_cast<double>(i) * 0.1);
	    single_feat.push_back(row);
	}

	glades::PCA pca_sf;
	ASSERT("PCA single feature: compute succeeds", pca_sf.compute(single_feat));
	ASSERT("PCA single feature: converged", pca_sf.converged());
	ASSERT("PCA single feature: 1 component", pca_sf.getNumComponents() == 1);

	const std::vector<double>& evals_sf = pca_sf.getEigenvalues();
	ASSERT("PCA single feature: 1 eigenvalue", evals_sf.size() == 1);
	ASSERT("PCA single feature: eigenvalue > 0", evals_sf[0] > 0.0);

	const std::vector<double>& ve_sf = pca_sf.getVarianceExplained();
	ASSERT("PCA single feature: variance = 100%", std::fabs(ve_sf[0] - 1.0) < 1e-6);

	// Lossless reconstruction
	const std::vector<std::vector<double> >& rdata_sf = pca_sf.getReconstructedData();
	double max_err_sf = 0.0;
	for (size_t i = 0; i < single_feat.size(); ++i)
	{
	    double err = std::fabs(rdata_sf[i][0] - single_feat[i][0]);
	    if (err > max_err_sf)
		max_err_sf = err;
	}
	ASSERT("PCA single feature: lossless reconstruction", max_err_sf < 1e-6);
    }

    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("PCA Edge Case: All-Zero Data\n");
    printf("-----------------------------------\n");

    {
	std::vector<std::vector<double> > zero_data(50, std::vector<double>(3, 0.0));

	glades::PCA pca_z;
	bool ok = pca_z.compute(zero_data);
	// All-zero data has zero variance; eigendecomposition should still converge
	// but all eigenvalues should be 0
	if (ok)
	{
	    ASSERT("PCA all-zero: converged", pca_z.converged());
	    const std::vector<double>& evals_z = pca_z.getEigenvalues();
	    for (size_t i = 0; i < evals_z.size(); ++i)
		ASSERT("PCA all-zero: eigenvalue == 0", std::fabs(evals_z[i]) < 1e-10);
	}
	printf("[PCA] All-zero test: compute returned %s\n", ok ? "true" : "false");
    }

    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("PCA Edge Case: Nearly Degenerate Eigenvalues\n");
    printf("-----------------------------------\n");

    {
	// Two features with nearly equal variance
	unsigned long lcg2 = 123;
	std::vector<std::vector<double> > degen_data;
	for (int i = 0; i < 200; ++i)
	{
	    lcg2 = lcg2 * 6364136223846793005ULL + 1442695040888963407ULL;
	    double n1 = (static_cast<double>(lcg2 >> 33) / 2147483648.0 - 0.5);
	    lcg2 = lcg2 * 6364136223846793005ULL + 1442695040888963407ULL;
	    double n2 = (static_cast<double>(lcg2 >> 33) / 2147483648.0 - 0.5);

	    std::vector<double> row;
	    row.push_back(n1);
	    row.push_back(n2 + n1 * 1e-10); // Nearly identical variance to first
	    degen_data.push_back(row);
	}

	glades::PCA pca_d;
	ASSERT("PCA degenerate: compute succeeds", pca_d.compute(degen_data));
	ASSERT("PCA degenerate: converged", pca_d.converged());

	const std::vector<double>& evals_d = pca_d.getEigenvalues();
	ASSERT("PCA degenerate: eigenvalues descending", evals_d[0] >= evals_d[1] - 1e-10);

	// Eigenvectors should still be orthogonal
	const std::vector<std::vector<double> > evecs_d = pca_d.getEigenvectors();
	double dot_d = 0.0;
	for (size_t k = 0; k < evecs_d[0].size(); ++k)
	    dot_d += evecs_d[0][k] * evecs_d[1][k];
	ASSERT("PCA degenerate: eigenvectors orthogonal", std::fabs(dot_d) < 1e-6);

	// Reconstruction should still be lossless
	const std::vector<std::vector<double> >& rdata_d = pca_d.getReconstructedData();
	double max_err_d = 0.0;
	for (size_t i = 0; i < degen_data.size(); ++i)
	{
	    for (size_t j = 0; j < degen_data[i].size(); ++j)
	    {
		double err = std::fabs(rdata_d[i][j] - degen_data[i][j]);
		if (err > max_err_d)
		    max_err_d = err;
	    }
	}
	ASSERT("PCA degenerate: lossless reconstruction", max_err_d < 1e-6);
    }

    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("PCA Edge Case: Large Condition Number\n");
    printf("-----------------------------------\n");

    {
	// One feature with huge variance, another with tiny variance
	unsigned long lcg3 = 456;
	std::vector<std::vector<double> > cond_data;
	for (int i = 0; i < 200; ++i)
	{
	    lcg3 = lcg3 * 6364136223846793005ULL + 1442695040888963407ULL;
	    double n1 = (static_cast<double>(lcg3 >> 33) / 2147483648.0 - 0.5);
	    lcg3 = lcg3 * 6364136223846793005ULL + 1442695040888963407ULL;
	    double n2 = (static_cast<double>(lcg3 >> 33) / 2147483648.0 - 0.5);

	    std::vector<double> row;
	    row.push_back(n1 * 1e6);   // Huge variance
	    row.push_back(n2 * 1e-6);  // Tiny variance
	    cond_data.push_back(row);
	}

	glades::PCA pca_c;
	ASSERT("PCA large cond: compute succeeds", pca_c.compute(cond_data));
	ASSERT("PCA large cond: converged", pca_c.converged());

	const std::vector<double>& evals_c = pca_c.getEigenvalues();
	ASSERT("PCA large cond: eigenvalues descending", evals_c[0] >= evals_c[1] - 1e-10);
	ASSERT("PCA large cond: first eigenvalue >> second", evals_c[0] > evals_c[1] * 1e6);

	// First PC should capture ~100% of variance
	const std::vector<double>& ve_c = pca_c.getVarianceExplained();
	ASSERT("PCA large cond: PC1 dominates", ve_c[0] > 0.999);

	// Eigenvectors should be orthogonal
	const std::vector<std::vector<double> > evecs_c = pca_c.getEigenvectors();
	double dot_c = 0.0;
	for (size_t k = 0; k < evecs_c[0].size(); ++k)
	    dot_c += evecs_c[0][k] * evecs_c[1][k];
	ASSERT("PCA large cond: eigenvectors orthogonal", std::fabs(dot_c) < 1e-6);

	// Reconstruction should be lossless
	const std::vector<std::vector<double> >& rdata_c = pca_c.getReconstructedData();
	double max_err_c = 0.0;
	for (size_t i = 0; i < cond_data.size(); ++i)
	{
	    for (size_t j = 0; j < cond_data[i].size(); ++j)
	    {
		double err = std::fabs(rdata_c[i][j] - cond_data[i][j]);
		if (err > max_err_c)
		    max_err_c = err;
	    }
	}
	// Relax tolerance for ill-conditioned case
	ASSERT("PCA large cond: reconstruction", max_err_c < 1e-2);
    }

    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("PCA Test: Error Returns\n");
    printf("-----------------------------------\n");

    {
	glades::PCA pca_err;

	// Empty data
	std::vector<std::vector<double> > empty_data;
	ASSERT("PCA error: empty data returns false", !pca_err.compute(empty_data));

	// Single sample
	std::vector<std::vector<double> > single_sample;
	std::vector<double> row;
	row.push_back(1.0);
	row.push_back(2.0);
	single_sample.push_back(row);
	ASSERT("PCA error: single sample returns false", !pca_err.compute(single_sample));

	// Zero features
	std::vector<std::vector<double> > zero_feat(5, std::vector<double>());
	ASSERT("PCA error: zero features returns false", !pca_err.compute(zero_feat));

	// Mismatched row lengths
	std::vector<std::vector<double> > mismatch;
	std::vector<double> r1;
	r1.push_back(1.0);
	r1.push_back(2.0);
	std::vector<double> r2;
	r2.push_back(3.0);
	mismatch.push_back(r1);
	mismatch.push_back(r2);
	ASSERT("PCA error: mismatched rows returns false", !pca_err.compute(mismatch));
    }

    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("PCA Test: fit() + transform() Separation\n");
    printf("-----------------------------------\n");

    {
	// Use the same y=x data
	std::vector<std::vector<double> > fit_data;
	for (int i = -100; i < 100; ++i)
	{
	    double x = static_cast<double>(i) / 100.0 * 5.0;
	    std::vector<double> pt;
	    pt.push_back(x);
	    pt.push_back(x);
	    fit_data.push_back(pt);
	}

	glades::PCA pca_fit;
	ASSERT("PCA fit: succeeds", pca_fit.fit(fit_data));
	ASSERT("PCA fit: converged", pca_fit.converged());
	ASSERT("PCA fit: has eigenvalues", pca_fit.getEigenvalues().size() == 2);

	// fit() should NOT populate transformed/reconstructed data
	ASSERT("PCA fit: no transformed data", pca_fit.getTransformedData().empty());
	ASSERT("PCA fit: no reconstructed data", pca_fit.getReconstructedData().empty());

	// But transform() should work
	std::vector<std::vector<double> > projected = pca_fit.transform(fit_data);
	ASSERT("PCA fit: transform works", projected.size() == fit_data.size());
	ASSERT("PCA fit: projected dims", !projected.empty() && projected[0].size() == 2);

	// And round-trip should be lossless
	std::vector<std::vector<double> > roundtrip = pca_fit.inverseTransform(projected);
	double max_rt_err = 0.0;
	for (size_t i = 0; i < fit_data.size(); ++i)
	{
	    for (size_t j = 0; j < fit_data[i].size(); ++j)
	    {
		double err = std::fabs(roundtrip[i][j] - fit_data[i][j]);
		if (err > max_rt_err)
		    max_rt_err = err;
	    }
	}
	ASSERT("PCA fit: round-trip lossless", max_rt_err < 1e-6);

	// fit() with reduced components
	glades::PCA pca_fit_k1;
	ASSERT("PCA fit k=1: succeeds", pca_fit_k1.fit(fit_data, 1));
	ASSERT("PCA fit k=1: 1 component", pca_fit_k1.getNumComponents() == 1);

	std::vector<std::vector<double> > proj_k1 = pca_fit_k1.transform(fit_data);
	ASSERT("PCA fit k=1: projected to 1 dim", !proj_k1.empty() && proj_k1[0].size() == 1);
    }

    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("PCA Test: partialFit() / finalizeFit()\n");
    printf("-----------------------------------\n");

    {
	// Generate data
	std::vector<std::vector<double> > full_data;
	unsigned long lcg4 = 789;
	for (int i = 0; i < 200; ++i)
	{
	    double x = static_cast<double>(i) / 200.0 * 10.0 - 5.0;
	    lcg4 = lcg4 * 6364136223846793005ULL + 1442695040888963407ULL;
	    double noise = (static_cast<double>(lcg4 >> 33) / 2147483648.0 - 0.5) * 0.1;
	    std::vector<double> pt;
	    pt.push_back(x);
	    pt.push_back(2.0 * x + noise);
	    pt.push_back(-x + 3.0 + noise * 0.5);
	    full_data.push_back(pt);
	}

	// Full compute for reference
	glades::PCA pca_full;
	ASSERT("PCA incremental ref: compute succeeds", pca_full.compute(full_data));

	// Incremental: feed in 4 batches of 50
	glades::PCA pca_inc;
	for (int batch = 0; batch < 4; ++batch)
	{
	    std::vector<std::vector<double> > batch_data(
		full_data.begin() + batch * 50,
		full_data.begin() + (batch + 1) * 50);
	    pca_inc.partialFit(batch_data);
	}
	ASSERT("PCA incremental: finalizeFit succeeds", pca_inc.finalizeFit());
	ASSERT("PCA incremental: converged", pca_inc.converged());

	// Compare eigenvalues with full compute
	const std::vector<double>& evals_full = pca_full.getEigenvalues();
	const std::vector<double>& evals_inc = pca_inc.getEigenvalues();
	ASSERT("PCA incremental: same number of eigenvalues",
	    evals_full.size() == evals_inc.size());

	double max_eval_diff = 0.0;
	for (size_t i = 0; i < evals_full.size(); ++i)
	{
	    double diff = std::fabs(evals_full[i] - evals_inc[i]);
	    if (diff > max_eval_diff)
		max_eval_diff = diff;
	}
	printf("[PCA] Incremental vs full eigenvalue max diff: %.2e\n", max_eval_diff);
	ASSERT("PCA incremental: eigenvalues match full", max_eval_diff < 1e-8);

	// Compare means
	const std::vector<double>& mean_full = pca_full.getMean();
	const std::vector<double>& mean_inc = pca_inc.getMean();
	double max_mean_diff = 0.0;
	for (size_t i = 0; i < mean_full.size(); ++i)
	{
	    double diff = std::fabs(mean_full[i] - mean_inc[i]);
	    if (diff > max_mean_diff)
		max_mean_diff = diff;
	}
	ASSERT("PCA incremental: means match full", max_mean_diff < 1e-10);

	// Transform with incremental model should produce same results
	std::vector<std::vector<double> > proj_full = pca_full.transform(full_data);
	std::vector<std::vector<double> > proj_inc = pca_inc.transform(full_data);
	double max_proj_diff = 0.0;
	for (size_t i = 0; i < proj_full.size(); ++i)
	{
	    for (size_t j = 0; j < proj_full[i].size(); ++j)
	    {
		double diff = std::fabs(std::fabs(proj_full[i][j]) - std::fabs(proj_inc[i][j]));
		if (diff > max_proj_diff)
		    max_proj_diff = diff;
	    }
	}
	// Compare absolute values since eigenvector signs can differ
	ASSERT("PCA incremental: projections match full", max_proj_diff < 1e-6);

	// Test partialFit with target components
	glades::PCA pca_inc2;
	pca_inc2.partialFit(full_data, 2);
	ASSERT("PCA incremental k=2: finalizeFit succeeds", pca_inc2.finalizeFit());
	ASSERT("PCA incremental k=2: 2 components", pca_inc2.getNumComponents() == 2);

	// finalizeFit with insufficient data should fail
	glades::PCA pca_inc_bad;
	ASSERT("PCA incremental: finalizeFit with no data fails", !pca_inc_bad.finalizeFit());
    }

    printf("============================================================\n");
}
