import React, { useState } from "react";
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [uploadResponse, setUploadResponse] = useState(null);
  const [edaResponse, setEdaResponse] = useState(null);
  const [survivalResponse, setSurvivalResponse] = useState(null);
  const [mlResponse, setMlResponse] = useState(null);
  // --- ADD STATE FOR CAUSAL RESPONSE ---
  const [causalResponse, setCausalResponse] = useState(null);
  
  const [loading, setLoading] = useState({
    upload: false,
    eda: false,
    survival: false,
    ml: false,
    causal: false // --- ADD LOADING STATE FOR CAUSAL ---
  });

  const API_BASE_URL = "http://localhost:8000";

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setUploadResponse(null);
    setEdaResponse(null);
    setSurvivalResponse(null);
    setMlResponse(null);
    setCausalResponse(null); // Reset on new file
  };

  const uploadFile = async () => {
    if (!file) return alert("Please select a CSV file first.");
    setLoading(prev => ({ ...prev, upload: true }));
    
    const formData = new FormData();
    formData.append("file", file);
    
    try {
      const res = await fetch(`${API_BASE_URL}/upload-csv/`, {
        method: "POST",
        body: formData,
      });
      
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Upload failed: ${res.status} - ${errorText}`);
      }
      
      const data = await res.json();
      setUploadResponse(data);
    } catch (err) {
      alert("Upload failed: " + err.message);
    } finally {
      setLoading(prev => ({ ...prev, upload: false }));
    }
  };

  const runEDA = async () => {
    if (!file || !uploadResponse) return;
    setLoading(prev => ({ ...prev, eda: true }));
    
    const reader = new FileReader();
    reader.onload = async () => {
      const base64Data = btoa(reader.result);
      try {
        const res = await fetch(`${API_BASE_URL}/run-eda/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            file: base64Data,
            churn_column: uploadResponse.churn_column,
            time_column: uploadResponse.time_column,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        setEdaResponse(data);
      } catch (err) {
        alert("EDA failed: " + err.message);
      } finally {
        setLoading(prev => ({ ...prev, eda: false }));
      }
    };
    reader.readAsBinaryString(file);
  };

  const runSurvival = async () => {
    if (!file || !uploadResponse || !uploadResponse.time_column) return;
    setLoading(prev => ({ ...prev, survival: true }));
    
    const reader = new FileReader();
    reader.onload = async () => {
      const base64Data = btoa(reader.result);
      try {
        const res = await fetch(`${API_BASE_URL}/run-survival/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            file: base64Data,
            churn_column: uploadResponse.churn_column,
            time_column: uploadResponse.time_column,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        setSurvivalResponse(data);
      } catch (err) {
        alert("Survival Analysis failed: " + err.message);
      } finally {
        setLoading(prev => ({ ...prev, survival: false }));
      }
    };
    reader.readAsBinaryString(file);
  };

  const runML = async () => {
    if (!file || !uploadResponse) return;
    setLoading(prev => ({ ...prev, ml: true }));
    
    const reader = new FileReader();
    reader.onload = async () => {
      const base64Data = btoa(reader.result);
      try {
        const res = await fetch(`${API_BASE_URL}/run-ml/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            file: base64Data,
            churn_column: uploadResponse.churn_column,
          }),
        });
        const data = await res.json();
        if(res.status !== 200) throw new Error(data.error);
        setMlResponse(data);
      } catch (err) {
        alert("ML Pipeline failed: " + err.message);
      } finally {
        setLoading(prev => ({ ...prev, ml: false }));
      }
    };
    reader.readAsBinaryString(file);
  };

  // --- NEW CAUSAL ANALYSIS FUNCTION ---
  const runCausal = async () => {
    if (!file || !uploadResponse) return;
    setLoading(prev => ({ ...prev, causal: true }));

    const reader = new FileReader();
    reader.onload = async () => {
      const base64Data = btoa(reader.result);
      try {
        const res = await fetch(`${API_BASE_URL}/run-causal/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            file: base64Data,
            churn_column: uploadResponse.churn_column,
          }),
        });
        const data = await res.json();
        if (res.status !== 200) throw new Error(data.error);
        setCausalResponse(data);
      } catch (err) {
        alert("Causal Analysis failed: " + err.message);
      } finally {
        setLoading(prev => ({ ...prev, causal: false }));
      }
    };
    reader.readAsBinaryString(file);
  };


  const downloadPDF = async () => {
    if (!edaResponse) return;
    const plots = [
      ...(edaResponse.distribution_plots || []),
      ...(edaResponse.correlation_plot ? [edaResponse.correlation_plot] : []),
      ...(edaResponse.churn_by_feature_plots || []),
    ];
    if (edaResponse.timeline_plot) {
      plots.push(edaResponse.timeline_plot);
    }
    try {
      const res = await fetch(`${API_BASE_URL}/generate-report/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plots: plots }),
      });
      
      if (!res.ok) throw new Error(await res.text());
      
      const data = await res.json();
      const link = document.createElement("a");
      link.href = `${API_BASE_URL}${data.pdf_url}`;
      link.download = "EDA_Report.pdf";
      link.click();
    } catch (err) {
      alert("Failed to generate PDF: " + err.message);
    }
  };

  // Helper to render missing values
  const renderMissingValues = (missing) => {
    if (!missing) return null;
    const hasCounts = missing.missing_counts && Object.keys(missing.missing_counts).length > 0;
    const hasPercentages = missing.missing_percentages && Object.keys(missing.missing_percentages).length > 0;

    return (
      <div>
        <h3 className="text-lg font-semibold mb-2 text-zinc-300">Missing Values</h3>
        {hasCounts && (
          <div className="mb-3">
            <h4 className="text-md font-medium mb-1 text-zinc-400">Missing Counts:</h4>
            <ul className="list-disc list-inside text-zinc-200 text-sm">
              {Object.entries(missing.missing_counts).map(([col, count]) => count > 0 && (
                <li key={col}><span className="font-medium">{col}</span>: {count}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-zinc-900 text-white px-4 py-10">
      <div className="max-w-4xl mx-auto space-y-10">
        <h1 className="text-5xl font-extrabold text-center text-cyan-400 tracking-tight">
          ChurnSense AI
        </h1>
        
        <div className="bg-zinc-800/70 backdrop-blur-md p-6 rounded-2xl shadow-lg border border-zinc-700">
          <label className="block mb-2 text-lg font-semibold text-zinc-200">
            Upload CSV Dataset
          </label>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="bg-zinc-700 text-white p-2 rounded w-full border border-zinc-600"
          />
          <button
            onClick={uploadFile}
            disabled={loading.upload}
            className="mt-4 w-full bg-cyan-500 hover:bg-cyan-600 text-white font-semibold py-2 rounded-lg transition-all disabled:opacity-50"
          >
            {loading.upload ? "Uploading..." : "Upload & Analyze"}
          </button>
        </div>

        {uploadResponse && (
          <div className="bg-zinc-800/70 backdrop-blur-md p-6 rounded-2xl shadow-lg border border-zinc-700">
            <h2 className="text-2xl font-semibold text-cyan-300 mb-2">Dataset Info</h2>
            <p>📊 Churn Column: <span className="font-bold">{uploadResponse.churn_column}</span></p>
            <p>⏳ Time Column: <span className="font-bold">{uploadResponse.time_column || "None detected"}</span></p>
            
            {/* --- UPDATED BUTTON GRID --- */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-4">
              <button onClick={runEDA} disabled={loading.eda} className="bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 rounded-lg transition-all disabled:opacity-50">
                {loading.eda ? "Running..." : "🧪 Run EDA"}
              </button>
              
              <button onClick={runSurvival} disabled={loading.survival || !uploadResponse.time_column} className="bg-pink-600 hover:bg-pink-700 text-white font-semibold py-2 rounded-lg transition-all disabled:opacity-50">
                {loading.survival ? "Running..." : "⏳ Survival"}
              </button>
              
              <button onClick={runML} disabled={loading.ml} className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2 rounded-lg transition-all disabled:opacity-50">
                {loading.ml ? "Training..." : "🤖 ML Prediction"}
              </button>

              <button onClick={runCausal} disabled={loading.causal} className="bg-orange-600 hover:bg-orange-700 text-white font-semibold py-2 rounded-lg transition-all disabled:opacity-50">
                {loading.causal ? "Analyzing..." : "🔎 Causal Analysis"}
              </button>
            </div>
          </div>
        )}

        {/* --- NEW CAUSAL ANALYSIS RESULTS SECTION --- */}
        {causalResponse && (
          <div className="bg-zinc-800/70 backdrop-blur-md p-6 rounded-2xl shadow-lg border border-zinc-700 space-y-6">
            <h2 className="text-2xl font-bold text-cyan-300">🔎 Causal Analysis Results</h2>
            {causalResponse.error ? (
              <p className="text-red-400">{causalResponse.error}</p>
            ) : (
              <>
                <div>
                  <h3 className="text-lg font-semibold mb-2 text-zinc-300">Causal Effect of Monthly Fee on Churn</h3>
                  <div className="bg-zinc-700/50 p-4 rounded-lg">
                    <p className="text-xl font-bold text-green-400">{Number(causalResponse.estimated_effect).toFixed(4)}</p>
                    <p className="text-zinc-400">This is the estimated change in the probability of churn for a one-unit increase in the monthly fee.</p>
                  </div>
                </div>
                <div>
                  <h3 className="text-lg font-semibold mb-2 text-zinc-300">Discovered Causal Graph</h3>
                  <img
                    src={`${API_BASE_URL}${causalResponse.causal_graph_url}`}
                    alt="Causal Graph"
                    className="rounded-lg w-full max-w-2xl border border-zinc-700 bg-white p-2"
                  />
                </div>
                <div>
                  <h3 className="text-lg font-semibold mb-2 text-zinc-300">Analysis Summary</h3>
                  <pre className="bg-zinc-900 p-4 rounded-lg text-xs text-zinc-300 overflow-x-auto">
                    {causalResponse.estimate_summary}
                  </pre>
                </div>
                 <div>
                  <h3 className="text-lg font-semibold mb-2 text-zinc-300">Model Refutation Results</h3>
                  <pre className="bg-zinc-900 p-4 rounded-lg text-xs text-zinc-300 overflow-x-auto">
                    {causalResponse.refutation_results}
                  </pre>
                </div>
              </>
            )}
          </div>
        )}

        {/* EDA Results */}
        {edaResponse && (
          <div className="bg-zinc-800/70 backdrop-blur-md p-6 rounded-2xl shadow-lg border border-zinc-700 space-y-6">
            <h2 className="text-2xl font-bold text-cyan-300">🧪 EDA Results</h2>
            {renderMissingValues(edaResponse.missing)}

            {edaResponse.distribution_plots && edaResponse.distribution_plots.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold mb-2 text-zinc-300">📈 Distribution Plots</h3>
                <div className="flex flex-wrap gap-4">
                  {edaResponse.distribution_plots.map((plot, i) => (
                    <img key={i} src={`${API_BASE_URL}${plot}`} alt={`Distribution ${i + 1}`} className="rounded-lg w-64 border border-zinc-700" />
                  ))}
                </div>
              </div>
            )}

            {edaResponse.correlation_plot && (
              <div>
                <h3 className="text-lg font-semibold mb-2 text-zinc-300">🔥 Correlation Heatmap</h3>
                <img src={`${API_BASE_URL}${edaResponse.correlation_plot}`} alt="Correlation Heatmap" className="rounded-lg w-full max-w-2xl border border-zinc-700" />
              </div>
            )}

            {edaResponse.churn_by_feature_plots && edaResponse.churn_by_feature_plots.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold mb-2 text-zinc-300">📊 Churn by Feature</h3>
                <div className="flex flex-wrap gap-4">
                  {edaResponse.churn_by_feature_plots.map((plot, i) => (
                    <img key={i} src={`${API_BASE_URL}${plot}`} alt={`Churn Feature ${i + 1}`} className="rounded-lg w-64 border border-zinc-700" />
                  ))}
                </div>
              </div>
            )}

            {edaResponse.timeline_plot && (
              <div>
                <h3 className="text-lg font-semibold mb-2 text-zinc-300">🕒 Churn Over Time</h3>
                <img src={`${API_BASE_URL}${edaResponse.timeline_plot}`} alt="Churn Timeline" className="rounded-lg w-full max-w-2xl border border-zinc-700" />
              </div>
            )}

            <button onClick={downloadPDF} className="mt-6 w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded-lg transition-all">
              📄 Download PDF Report
            </button>
          </div>
        )}

        {/* Survival Analysis Results */}
        {survivalResponse && (
          <div className="bg-zinc-800/70 backdrop-blur-md p-6 rounded-2xl shadow-lg border border-zinc-700 space-y-6">
            <h2 className="text-2xl font-bold text-cyan-300">⏳ Survival Analysis Results</h2>
            {survivalResponse.kaplan_meier_plot && (
                <div>
                <h3 className="text-lg font-semibold mb-2 text-zinc-300">📉 Kaplan–Meier Survival Curve</h3>
                <img src={`${API_BASE_URL}${survivalResponse.kaplan_meier_plot}`} alt="Kaplan Meier" className="rounded-lg w-full max-w-2xl border border-zinc-700" />
                </div>
            )}
            {survivalResponse.cox_model_plot && (
                <div>
                <h3 className="text-lg font-semibold mb-2 text-zinc-300">⚠ Cox Proportional Hazards Model</h3>
                <img src={`${API_BASE_URL}${survivalResponse.cox_model_plot}`} alt="Cox Model" className="rounded-lg w-full max-w-2xl border border-zinc-700" />
                </div>
            )}
            {survivalResponse.risk_scores && (
                <div>
                <h3 className="text-lg font-semibold mb-2 text-zinc-300">🏆 Top 10 High-Risk Customers</h3>
                <ul className="list-disc list-inside text-zinc-200">
                    {Object.entries(survivalResponse.risk_scores).map(([customer, score]) => (
                    <li key={customer}>Customer {customer}: Risk Score {Number(score).toFixed(3)}</li>
                    ))}
                </ul>
                </div>
            )}
          </div>
        )}

        {/* ML Results */}
        {mlResponse && (
          <div className="bg-zinc-800/70 backdrop-blur-md p-6 rounded-2xl shadow-lg border border-zinc-700 space-y-6">
            <h2 className="text-2xl font-bold text-cyan-300">🤖 Machine Learning Results</h2>
            {mlResponse.error ? <div className="text-red-400">Error: {mlResponse.error}</div> : (
              <>
                {mlResponse.best_model && (
                  <div className="bg-zinc-700/50 p-4 rounded-lg">
                    <h3 className="text-lg font-semibold mb-2 text-green-400">🏆 Best Model: {mlResponse.best_model.name}</h3>
                    {mlResponse.best_model.metrics && (
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
                        <div>Accuracy: <span className="font-bold">{(mlResponse.best_model.metrics.accuracy * 100).toFixed(1)}%</span></div>
                        <div>Precision: <span className="font-bold">{(mlResponse.best_model.metrics.precision * 100).toFixed(1)}%</span></div>
                        <div>Recall: <span className="font-bold">{(mlResponse.best_model.metrics.recall * 100).toFixed(1)}%</span></div>
                        <div>F1-Score: <span className="font-bold">{(mlResponse.best_model.metrics.f1 * 100).toFixed(1)}%</span></div>
                        <div>ROC-AUC: <span className="font-bold">{mlResponse.best_model.metrics.roc_auc.toFixed(3)}</span></div>
                      </div>
                    )}
                  </div>
                )}
                {mlResponse.plots && (
                  <div>
                    <h3 className="text-lg font-semibold mb-2 text-zinc-300">📊 Model Comparison</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {mlResponse.plots.comparison && <img src={`${API_BASE_URL}${mlResponse.plots.comparison}`} alt="Model Comparison" className="rounded-lg border border-zinc-700" />}
                      {mlResponse.plots.roc_curves && <img src={`${API_BASE_URL}${mlResponse.plots.roc_curves}`} alt="ROC Curves" className="rounded-lg border border-zinc-700" />}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
