import React, { useState } from "react";
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [uploadResponse, setUploadResponse] = useState(null);
  const [edaResponse, setEdaResponse] = useState(null);
  const [survivalResponse, setSurvivalResponse] = useState(null);
  const [mlResponse, setMlResponse] = useState(null);
  const [causalResponse, setCausalResponse] = useState(null);
  const [rlLogs, setRlLogs] = useState("");
  const [rlRecommendation, setRlRecommendation] = useState("");
  const [rlUserState, setRlUserState] = useState({ age: 30, fee: 150, activity: 25 });
  
  // This new state will control which analysis result is visible.
  const [activeAnalysis, setActiveAnalysis] = useState(null);
  
  const [loading, setLoading] = useState({
    upload: false,
    eda: false,
    survival: false,
    ml: false,
    causal: false,
    rlTraining: false,
    rlRecommend: false,
  });

  const API_BASE_URL = "http://localhost:8000";

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    // Reset all states when a new file is chosen
    setUploadResponse(null);
    setEdaResponse(null);
    setSurvivalResponse(null);
    setMlResponse(null);
    setCausalResponse(null);
    setRlLogs("");
    setRlRecommendation("");
    setActiveAnalysis(null); // Reset the active view
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
        setActiveAnalysis('eda'); // Switch view to EDA results
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
        setActiveAnalysis('survival'); // Switch view to Survival results
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
        setActiveAnalysis('ml'); // Switch view to ML results
      } catch (err) {
        alert("ML Pipeline failed: " + err.message);
      } finally {
        setLoading(prev => ({ ...prev, ml: false }));
      }
    };
    reader.readAsBinaryString(file);
  };

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
        setActiveAnalysis('causal'); // Switch view to Causal results
      } catch (err) {
        alert("Causal Analysis failed: " + err.message);
      } finally {
        setLoading(prev => ({ ...prev, causal: false }));
      }
    };
    reader.readAsBinaryString(file);
  };
  
  const runRLTraining = () => {
    if (!uploadResponse) return alert("Please upload a file first.");
    setLoading(prev => ({ ...prev, rlTraining: true }));
    setRlLogs(''); 
    setRlRecommendation('');
    setActiveAnalysis('rl'); // Switch view to RL results
    
    const eventSource = new EventSource(`${API_BASE_URL}/train-rl/`);
    
    eventSource.addEventListener('message', (event) => {
        setRlLogs(prevLogs => prevLogs + event.data + '\n');
    });

    eventSource.addEventListener('close', () => {
        eventSource.close();
        setLoading(prev => ({ ...prev, rlTraining: false }));
    });
    
    eventSource.onerror = () => {
        setRlLogs(prevLogs => prevLogs + 'Error: Connection to server failed or stream ended.\n');
        eventSource.close();
        setLoading(prev => ({ ...prev, rlTraining: false }));
    };
  };

  const handleRLInputChange = (e) => {
    const { name, value } = e.target;
    setRlUserState(prevState => ({ ...prevState, [name]: parseFloat(value) || 0 }));
  };

  const getRLRecommendation = async () => {
    setLoading(prev => ({ ...prev, rlRecommend: true }));
    setRlRecommendation('Getting recommendation...');
    try {
        const response = await fetch(`${API_BASE_URL}/get-rl-recommendation/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(rlUserState)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'An error occurred.');
        setRlRecommendation(data.recommendation);
    } catch (error) {
        setRlRecommendation(`Error: ${error.message}`);
    } finally {
      setLoading(prev => ({ ...prev, rlRecommend: false }));
    }
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

  const renderMissingValues = (missing) => {
    if (!missing) return null;
    const hasCounts = missing.missing_counts && Object.keys(missing.missing_counts).length > 0;
    
    if (!hasCounts || Object.values(missing.missing_counts).every(c => c === 0)) {
        return (
            <div>
                <h3 className="text-lg font-semibold mb-2 text-zinc-300">Missing Values</h3>
                <p className="text-green-400">✅ No missing values found in the dataset.</p>
            </div>
        )
    }

    return (
      <div>
        <h3 className="text-lg font-semibold mb-2 text-zinc-300">Missing Values</h3>
        <div className="mb-3">
          <h4 className="text-md font-medium mb-1 text-zinc-400">Missing Counts:</h4>
          <ul className="list-disc list-inside text-zinc-200 text-sm">
            {Object.entries(missing.missing_counts).map(([col, count]) => count > 0 && (
              <li key={col}><span className="font-medium">{col}</span>: {count}</li>
            ))}
          </ul>
        </div>
      </div>
    );
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>ChurnSense AI</h1>
      </header>
      
      <div className="tab-content">
          <label className="block mb-2 text-lg font-semibold text-zinc-200">
            Upload CSV Dataset
          </label>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="w-full"
          />
          <button
            onClick={uploadFile}
            disabled={loading.upload}
            className="mt-4 w-full bg-cyan-500"
          >
            {loading.upload ? "Uploading..." : "Upload & Analyze"}
          </button>
      </div>

      {uploadResponse && (
        <div className="tab-content">
          <h2 className="text-2xl font-semibold text-cyan-300 mb-2">Dataset Info & Actions</h2>
          <p>📊 Churn Column: <span className="font-bold">{uploadResponse.churn_column}</span></p>
          <p>⏳ Time Column: <span className="font-bold">{uploadResponse.time_column || "None detected"}</span></p>
          
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mt-4">
            <button onClick={runEDA} disabled={loading.eda} className="bg-purple-600">
              {loading.eda ? "Running..." : "🧪 Run EDA"}
            </button>
            
            <button onClick={runSurvival} disabled={loading.survival || !uploadResponse.time_column} className="bg-pink-600">
              {loading.survival ? "Running..." : "⏳ Survival"}
            </button>
            
            <button onClick={runML} disabled={loading.ml} className="bg-green-600">
              {loading.ml ? "Training..." : "🤖 ML Prediction"}
            </button>

            <button onClick={runCausal} disabled={loading.causal} className="bg-orange-600">
              {loading.causal ? "Analyzing..." : "🔎 Causal Analysis"}
            </button>

            <button onClick={runRLTraining} disabled={loading.rlTraining} className="bg-teal-600">
              {loading.rlTraining ? "Training..." : "🧠 Train RL Agent"}
            </button>
          </div>
        </div>
      )}

      {/* --- This area now only shows the active analysis result --- */}

      {activeAnalysis === 'eda' && edaResponse && (
         <div className="tab-content space-y-6">
          <h2 className="text-2xl font-bold text-cyan-300">🧪 EDA Results</h2>
          {renderMissingValues(edaResponse.missing)}
          {edaResponse.distribution_plots && edaResponse.distribution_plots.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold mb-2 text-zinc-300">📈 Distribution Plots</h3>
              <div className="flex flex-wrap gap-4">
                {edaResponse.distribution_plots.map((plot, i) => (
                  <img key={i} src={`${API_BASE_URL}${plot}`} alt={`Distribution ${i + 1}`} className="w-64" />
                ))}
              </div>
            </div>
          )}
          {edaResponse.correlation_plot && (
            <div>
              <h3 className="text-lg font-semibold mb-2 text-zinc-300">🔥 Correlation Heatmap</h3>
              <img src={`${API_BASE_URL}${edaResponse.correlation_plot}`} alt="Correlation Heatmap" className="max-w-2xl" />
            </div>
          )}
          {/* --- FIX: ADDED THE BUTTON BACK --- */}
          <button onClick={downloadPDF} className="mt-6 w-full bg-blue-600">
            📄 Download PDF Report
          </button>
         </div>
      )}

      {activeAnalysis === 'survival' && survivalResponse && (
         <div className="tab-content space-y-6">
          <h2 className="text-2xl font-bold text-cyan-300">⏳ Survival Analysis Results</h2>
          {survivalResponse.kaplan_meier_plot && (
              <div>
              <h3 className="text-lg font-semibold mb-2 text-zinc-300">📉 Kaplan–Meier Survival Curve</h3>
              <img src={`${API_BASE_URL}${survivalResponse.kaplan_meier_plot}`} alt="Kaplan Meier" className="max-w-2xl" />
              </div>
          )}
          {survivalResponse.cox_model_plot && (
              <div>
              <h3 className="text-lg font-semibold mb-2 text-zinc-300">⚠ Cox Proportional Hazards Model</h3>
              <img src={`${API_BASE_URL}${survivalResponse.cox_model_plot}`} alt="Cox Model" className="max-w-2xl" />
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

      {activeAnalysis === 'ml' && mlResponse && (
         <div className="tab-content space-y-6">
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
                    {mlResponse.plots.comparison && <img src={`${API_BASE_URL}${mlResponse.plots.comparison}`} alt="Model Comparison" />}
                    {mlResponse.plots.roc_curves && <img src={`${API_BASE_URL}${mlResponse.plots.roc_curves}`} alt="ROC Curves" />}
                  </div>
                </div>
              )}
            </>
          )}
         </div>
      )}

      {activeAnalysis === 'causal' && causalResponse && (
          <div className="tab-content space-y-6">
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
                  className="w-full max-w-2xl bg-white p-2"
                />
              </div>
            </>
          )}
        </div>
      )}

      {activeAnalysis === 'rl' && (
        <div className="tab-content space-y-6">
          <h2 className="text-2xl font-bold text-cyan-300">🧠 Reinforcement Learning</h2>
          <div className="rl-container">
            <div className="rl-section">
              <h3>Training Log</h3>
              <pre className="log-box">{loading.rlTraining && !rlLogs ? 'Starting...' : rlLogs}</pre>
            </div>
            <div className="rl-section">
              <h3>Get Action Recommendation</h3>
              <div className="recommendation-form">
                  <label>Age: <input type="number" name="age" value={rlUserState.age} onChange={handleRLInputChange} /></label>
                  <label>Fee: <input type="number" name="fee" value={rlUserState.fee} onChange={handleRLInputChange} /></label>
                  <label>Activity: <input type="number" name="activity" value={rlUserState.activity} onChange={handleRLInputChange} /></label>
                  <button onClick={getRLRecommendation} disabled={loading.rlRecommend} className="bg-blue-600">
                    {loading.rlRecommend ? "..." : "Get Action"}
                  </button>
              </div>
              {rlRecommendation && (
                  <div className="recommendation-result">
                      <p>{rlRecommendation}</p>
                  </div>
              )}
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

export default App;
