import React, { useState, useMemo, useCallback } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell, Legend, LineChart, Line } from 'recharts';

const colorPalette = [
  // Row 1: Primary distinct colors (blues, reds, greens)
  "#2563eb", "#dc2626", "#059669", "#ea580c", "#0891b2",
  // Row 2: Purples, indigos, teals
  "#7c3aed", "#4f46e5", "#14b8a6", "#9333ea", "#6366f1",
  // Row 3: Dark variants for contrast
  "#1e40af", "#b91c1c", "#047857", "#9a3412", "#0e7490",
  // Row 4: Medium variants
  "#6d28d9", "#4338ca", "#0d9488", "#a21caf", "#0369a1",
  // Row 5: Lighter variants (still visible)
  "#3b82f6", "#ef4444", "#10b981", "#f97316", "#06b6d4",
  // Row 6: Additional distinct colors
  "#8b5cf6", "#ec4899", "#84cc16", "#eab308", "#22c55e"
];

const parseCSV = (csvString, fileName) => {
  const lines = csvString.trim().split('\n');
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));

  return lines.slice(1).map((line) => {
    const values = [];
    let current = '';
    let inQuotes = false;
    for (let char of line) {
      if (char === '"') inQuotes = !inQuotes;
      else if (char === ',' && !inQuotes) { values.push(current.trim()); current = ''; }
      else current += char;
    }
    values.push(current.trim());

    const obj = { _source: fileName };
    headers.forEach((header, i) => {
      let val = values[i] || '';
      val = val.replace(/"/g, '');
      const num = parseFloat(val);
      obj[header] = isNaN(num) ? val : num;
    });
    return obj;
  });
};

// Detect dataset type based on columns
const detectDatasetType = (data) => {
  if (data.length === 0) return 'unknown';
  const sample = data[0];
  
  // Check for speed-only dataset (multiple column name variants)
  // Old format: prefill_speed, decode_speed
  // New format: prefill_tps, decoding_tps
  const hasPrefillSpeed = sample.prefill_speed !== undefined || sample.prefill_tps !== undefined;
  const hasDecodeSpeed = sample.decode_speed !== undefined || sample.decoding_tps !== undefined;
  const hasQualityMetrics = sample.accuracy !== undefined || sample.rougeL !== undefined || sample.rouge1 !== undefined;
  
  if (hasPrefillSpeed && hasDecodeSpeed && !hasQualityMetrics) {
    return 'speed';
  }
  
  // Check for ROUGE scores (LongBench)
  if (sample.rouge1 !== undefined || sample.rougeL !== undefined) {
    return 'longbench';
  }
  
  // Check for accuracy/BLEURT (TruthfulQA)
  if (sample.accuracy !== undefined) {
    return 'truthfulqa';
  }
  
  return 'unknown';
};

const processDataTruthfulQA = (data) => {
  if (data.length === 0) return [];

  const valid = data.filter(d => d.accuracy > 0 && (d.avg_decode_speed > 0 || d.decode_speed > 0));
  if (valid.length === 0) return [];

  valid.forEach(d => {
    if (!d.decode_speed && d.avg_decode_speed) d.decode_speed = d.avg_decode_speed;
    if (!d.prefill_speed && d.avg_prefill_speed) d.prefill_speed = d.avg_prefill_speed;
    if (!d.total_speed && d.avg_total_speed) d.total_speed = d.avg_total_speed;
  });

  const hasBleurt = valid.some(d => d.bleurt_score !== undefined && !isNaN(d.bleurt_score));

  const minAcc = Math.min(...valid.map(d => d.accuracy));
  const maxAcc = Math.max(...valid.map(d => d.accuracy));
  const minSpeed = Math.min(...valid.map(d => d.decode_speed));
  const maxSpeed = Math.max(...valid.map(d => d.decode_speed));

  let minBleurt = 0, maxBleurt = 0;
  if (hasBleurt) {
    const bleurtValues = valid.filter(d => d.bleurt_score !== undefined && !isNaN(d.bleurt_score)).map(d => d.bleurt_score);
    minBleurt = Math.min(...bleurtValues);
    maxBleurt = Math.max(...bleurtValues);
  }

  const accRange = maxAcc - minAcc || 1;
  const speedRange = maxSpeed - minSpeed || 1;
  const bleurtRange = maxBleurt - minBleurt || 1;

  return valid.map(d => {
    const norm_accuracy = (d.accuracy - minAcc) / accRange;
    const norm_speed = (d.decode_speed - minSpeed) / speedRange;
    const norm_bleurt = hasBleurt && d.bleurt_score !== undefined && !isNaN(d.bleurt_score)
      ? (d.bleurt_score - minBleurt) / bleurtRange : null;

    const combined = norm_bleurt !== null
      ? (norm_accuracy + norm_speed + norm_bleurt) / 3
      : (norm_accuracy + norm_speed) / 2;

    return {
      ...d,
      norm_accuracy, norm_speed, norm_bleurt, combined,
      hasBleurt: norm_bleurt !== null,
      model_short: d.model ? d.model.replace(/-Q4_K_M\.gguf|\.gguf|_q4_0\.gguf/g, '').substring(0, 25) : 'unknown'
    };
  });
};

const processDataLongBench = (data) => {
  if (data.length === 0) return [];

  // Use total_speed if available, otherwise avg_total_speed_tok_s, or decode_speed variants
  const valid = data.filter(d => {
    const hasSpeed = d.avg_total_speed_tok_s > 0 || d.total_speed > 0 || d.decode_speed > 0 || d.avg_decode_speed > 0 || d.avg_total_speed > 0;
    const hasRouge = d.rougeL !== undefined && d.rougeL > 0;
    return hasSpeed && hasRouge;
  });
  
  if (valid.length === 0) return [];

  // Normalize column names for backwards compatibility
  valid.forEach(d => {
    // Handle speed column variants
    if (!d.total_speed && d.avg_total_speed_tok_s) d.total_speed = d.avg_total_speed_tok_s;
    if (!d.decode_speed) {
      d.decode_speed = d.avg_decode_speed || d.total_speed || d.avg_total_speed_tok_s || d.avg_total_speed || 0;
    }
    
    // Handle runtime column variants
    if (!d.runtime_seconds) {
      d.runtime_seconds = d.runtime_sec || d.runtime || 0;
    }
    
    // StreamLLM backwards compatibility - set to undefined if not present
    if (d.streamllm_enabled === undefined || d.streamllm_enabled === '') {
      d.streamllm_enabled = undefined;
    }
    if (d.sink_count === undefined || d.sink_count === '') {
      d.sink_count = undefined;
    }
    if (d.sink_bias === undefined || d.sink_bias === '') {
      d.sink_bias = undefined;
    }
  });

  const minRougeL = Math.min(...valid.map(d => d.rougeL));
  const maxRougeL = Math.max(...valid.map(d => d.rougeL));
  const minSpeed = Math.min(...valid.map(d => d.decode_speed));
  const maxSpeed = Math.max(...valid.map(d => d.decode_speed));
  
  const minRouge1 = Math.min(...valid.map(d => d.rouge1 || 0));
  const maxRouge1 = Math.max(...valid.map(d => d.rouge1 || 0));

  const rougeLRange = maxRougeL - minRougeL || 1;
  const speedRange = maxSpeed - minSpeed || 1;
  const rouge1Range = maxRouge1 - minRouge1 || 1;

  return valid.map(d => {
    const norm_rougeL = (d.rougeL - minRougeL) / rougeLRange;
    const norm_speed = (d.decode_speed - minSpeed) / speedRange;
    const norm_rouge1 = (d.rouge1 - minRouge1) / rouge1Range;

    const combined = (norm_rougeL + norm_speed + norm_rouge1) / 3;

    return {
      ...d,
      norm_rougeL, norm_speed, norm_rouge1, combined,
      model_short: d.model ? d.model.replace(/-Q4_K_M\.gguf|\.gguf|_q4_0\.gguf/g, '').substring(0, 25) : 'unknown'
    };
  });
};

const processDataSpeed = (data) => {
  if (data.length === 0) return [];

  // Normalize column names to support both formats
  const normalized = data.map(d => ({
    ...d,
    prefill_speed: d.prefill_speed || d.prefill_tps || 0,
    decode_speed: d.decode_speed || d.decoding_tps || 0
  }));

  const valid = normalized.filter(d => {
    const hasPrefill = d.prefill_speed > 0;
    const hasDecode = d.decode_speed > 0;
    return hasPrefill && hasDecode;
  });
  
  if (valid.length === 0) return [];

  const minPrefill = Math.min(...valid.map(d => d.prefill_speed));
  const maxPrefill = Math.max(...valid.map(d => d.prefill_speed));
  const minDecode = Math.min(...valid.map(d => d.decode_speed));
  const maxDecode = Math.max(...valid.map(d => d.decode_speed));

  const prefillRange = maxPrefill - minPrefill || 1;
  const decodeRange = maxDecode - minDecode || 1;

  return valid.map(d => {
    const norm_prefill = (d.prefill_speed - minPrefill) / prefillRange;
    const norm_decode = (d.decode_speed - minDecode) / decodeRange;

    // Combined speed score (weighted average: decode more important)
    const combined = (norm_prefill * 0.3 + norm_decode * 0.7);

    return {
      ...d,
      norm_prefill, 
      norm_decode, 
      combined,
      model_short: d.model ? d.model.substring(0, 30) : 'unknown'
    };
  });
};

const aggregateByParam = (data, param, datasetType) => {
  if (data.length === 0 || data[0][param] === undefined) return [];
  const groups = {};
  data.forEach(d => {
    const key = d[param];
    if (key === undefined || key === '') return;
    if (!groups[key]) {
      groups[key] = { 
        [param]: key, 
        combined: [], 
        speed: [], 
        runtime: [],
        // TruthfulQA specific
        accuracy: [], 
        bleurt: [],
        prefill: [],
        // LongBench specific
        rouge1: [],
        rouge2: [],
        rougeL: [],
        rougeLsum: []
      };
    }
    groups[key].combined.push(d.combined);
    groups[key].speed.push(d.decode_speed);
    if (d.runtime_seconds) groups[key].runtime.push(d.runtime_seconds);
    
    if (datasetType === 'truthfulqa') {
      groups[key].accuracy.push(d.accuracy);
      if (d.bleurt_score !== undefined && !isNaN(d.bleurt_score)) groups[key].bleurt.push(d.bleurt_score);
      if (d.prefill_speed) groups[key].prefill.push(d.prefill_speed);
    } else if (datasetType === 'longbench') {
      if (d.rouge1) groups[key].rouge1.push(d.rouge1);
      if (d.rouge2) groups[key].rouge2.push(d.rouge2);
      if (d.rougeL) groups[key].rougeL.push(d.rougeL);
      if (d.rougeLsum) groups[key].rougeLsum.push(d.rougeLsum);
    }
  });
  
  return Object.values(groups).map(g => {
    const result = {
      [param]: g[param],
      combined: g.combined.reduce((a,b) => a+b, 0) / g.combined.length,
      speed: g.speed.reduce((a,b) => a+b, 0) / g.speed.length,
      runtime: g.runtime.length > 0 ? g.runtime.reduce((a,b) => a+b, 0) / g.runtime.length : null,
      count: g.combined.length
    };
    
    if (datasetType === 'truthfulqa') {
      result.accuracy = g.accuracy.reduce((a,b) => a+b, 0) / g.accuracy.length;
      result.bleurt = g.bleurt.length > 0 ? g.bleurt.reduce((a,b) => a+b, 0) / g.bleurt.length : null;
      result.prefill = g.prefill.length > 0 ? g.prefill.reduce((a,b) => a+b, 0) / g.prefill.length : null;
    } else if (datasetType === 'longbench') {
      result.rouge1 = g.rouge1.length > 0 ? g.rouge1.reduce((a,b) => a+b, 0) / g.rouge1.length : null;
      result.rouge2 = g.rouge2.length > 0 ? g.rouge2.reduce((a,b) => a+b, 0) / g.rouge2.length : null;
      result.rougeL = g.rougeL.length > 0 ? g.rougeL.reduce((a,b) => a+b, 0) / g.rougeL.length : null;
      result.rougeLsum = g.rougeLsum.length > 0 ? g.rougeLsum.reduce((a,b) => a+b, 0) / g.rougeLsum.length : null;
    }
    
    return result;
  }).sort((a,b) => {
    const aVal = a[param], bVal = b[param];
    if (typeof aVal === 'number') return aVal - bVal;
    return String(aVal).localeCompare(String(bVal));
  });
};

const CustomTooltipTruthfulQA = ({ active, payload, modelColors }) => {
  if (active && payload && payload.length) {
    const d = payload[0].payload;
    return (
      <div style={{ background: 'rgba(15, 23, 42, 0.95)', border: '1px solid rgba(148, 163, 184, 0.3)', borderRadius: '8px', padding: '12px 16px', fontSize: '12px', color: '#111827', boxShadow: '0 4px 20px rgba(0,0,0,0.4)', maxWidth: '380px' }}>
        <div style={{ fontWeight: 700, marginBottom: 8, color: modelColors?.[d.model_short] || '#fff', fontSize: '13px' }}>
          Run #{d.run_id} • {d.model_short}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '4px 12px' }}>
          <span style={{ color: '#374151' }}>Accuracy:</span><span style={{ fontWeight: 600 }}>{(d.accuracy * 100).toFixed(1)}%</span><span></span>
          <span style={{ color: '#374151' }}>Decode:</span><span style={{ fontWeight: 600 }}>{d.decode_speed?.toFixed(1)} tok/s</span><span></span>
          {d.prefill_speed && <><span style={{ color: '#374151' }}>Prefill:</span><span>{d.prefill_speed?.toFixed(1)} tok/s</span><span></span></>}
          {d.bleurt_score !== undefined && !isNaN(d.bleurt_score) && <><span style={{ color: '#374151' }}>BLEURT:</span><span style={{ color: '#06b6d4' }}>{d.bleurt_score?.toFixed(3)}</span><span></span></>}
          <span style={{ color: '#374151' }}>Combined:</span><span style={{ fontWeight: 600, color: '#fbbf24' }}>{d.combined?.toFixed(3)}</span><span></span>
          {d.temperature !== undefined && <><span style={{ color: '#374151' }}>Temp:</span><span>{d.temperature}</span><span></span></>}
          {d.top_k !== undefined && <><span style={{ color: '#374151' }}>Top-K:</span><span>{d.top_k}</span><span></span></>}
          {d.ctx_size !== undefined && <><span style={{ color: '#374151' }}>Ctx:</span><span>{d.ctx_size}</span><span></span></>}
          {d.runtime_seconds !== undefined && <><span style={{ color: '#374151' }}>Runtime:</span><span>{d.runtime_seconds}s</span><span></span></>}
        </div>
      </div>
    );
  }
  return null;
};

const CustomTooltipLongBench = ({ active, payload, modelColors }) => {
  if (active && payload && payload.length) {
    const d = payload[0].payload;
    return (
      <div style={{ background: 'rgba(15, 23, 42, 0.95)', border: '1px solid rgba(148, 163, 184, 0.3)', borderRadius: '8px', padding: '12px 16px', fontSize: '12px', color: '#111827', boxShadow: '0 4px 20px rgba(0,0,0,0.4)', maxWidth: '380px' }}>
        <div style={{ fontWeight: 700, marginBottom: 8, color: modelColors?.[d.model_short] || '#fff', fontSize: '13px' }}>
          Run #{d.run_id} • {d.model_short}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '4px 12px' }}>
          <span style={{ color: '#374151' }}>ROUGE-L:</span><span style={{ fontWeight: 600, color: '#8b5cf6' }}>{d.rougeL?.toFixed(4)}</span><span></span>
          <span style={{ color: '#374151' }}>ROUGE-1:</span><span style={{ fontWeight: 600 }}>{d.rouge1?.toFixed(4)}</span><span></span>
          <span style={{ color: '#374151' }}>ROUGE-2:</span><span>{d.rouge2?.toFixed(4)}</span><span></span>
          <span style={{ color: '#374151' }}>Speed:</span><span style={{ fontWeight: 600, color: '#059669' }}>{d.decode_speed?.toFixed(1)} tok/s</span><span></span>
          <span style={{ color: '#374151' }}>Combined:</span><span style={{ fontWeight: 600, color: '#fbbf24' }}>{d.combined?.toFixed(3)}</span><span></span>
          {d.temperature !== undefined && <><span style={{ color: '#374151' }}>Temp:</span><span>{d.temperature}</span><span></span></>}
          {d.ctx_size !== undefined && <><span style={{ color: '#374151' }}>Ctx:</span><span>{d.ctx_size}</span><span></span></>}
          {d.token_limit !== undefined && <><span style={{ color: '#374151' }}>Tokens:</span><span>{d.token_limit}</span><span></span></>}
          {d.streamllm_enabled !== undefined && <><span style={{ color: '#374151' }}>StreamLLM:</span><span>{d.streamllm_enabled}</span><span></span></>}
          {d.sink_count !== undefined && d.streamllm_enabled && <><span style={{ color: '#374151' }}>Sinks:</span><span>{d.sink_count}</span><span></span></>}
          {d.runtime_seconds !== undefined && <><span style={{ color: '#374151' }}>Runtime:</span><span>{d.runtime_seconds}s</span><span></span></>}
        </div>
      </div>
    );
  }
  return null;
};

const ScatterLegend = ({ modelColors, models }) => {
  if (!models || models.length === 0) return null;
  
  return (
    <div style={{ 
      marginTop: 12, 
      padding: '8px 12px', 
      background: '#f9fafb', 
      border: '1px solid #e5e7eb', 
      borderRadius: 8,
      fontSize: '0.75rem'
    }}>
      <div style={{ fontWeight: 600, marginBottom: 6, color: '#374151' }}>Models:</div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
        {models.map(model => (
          <div key={model} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{ 
              width: 10, 
              height: 10, 
              borderRadius: '50%', 
              background: modelColors[model] || '#6b7280',
              border: '1px solid #374151'
            }} />
            <span style={{ color: '#374151' }}>{model}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

const ParamChart = ({ data, param, title, datasetType }) => {
  if (!data || data.length < 2) return null;
  
  const hasBleurt = datasetType === 'truthfulqa' && data.some(d => d.bleurt !== null);
  const hasRouge = datasetType === 'longbench';

  return (
    <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 24 }}>
      <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 16, color: '#374151' }}>{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey={param} stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
          <YAxis yAxisId="left" stroke="#10b981" tick={{ fill: '#10b981', fontSize: 11 }} />
          <YAxis yAxisId="right" orientation="right" stroke="#8b5cf6" tick={{ fill: '#8b5cf6', fontSize: 11 }} domain={[0, 1]} />
          <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8, fontSize: 12 }} />
          <Legend />
          <Line yAxisId="left" type="monotone" dataKey="speed" name="Speed (tok/s)" stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} />
          
          {datasetType === 'truthfulqa' && (
            <>
              <Line yAxisId="right" type="monotone" dataKey="accuracy" name="Accuracy" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 4 }} />
              {hasBleurt && <Line yAxisId="right" type="monotone" dataKey="bleurt" name="BLEURT (norm)" stroke="#06b6d4" strokeWidth={2} dot={{ r: 3 }} strokeDasharray="5 5" />}
            </>
          )}
          
          {hasRouge && (
            <>
              <Line yAxisId="right" type="monotone" dataKey="rougeL" name="ROUGE-L" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 4 }} />
              <Line yAxisId="right" type="monotone" dataKey="rouge1" name="ROUGE-1" stroke="#ec4899" strokeWidth={1.5} dot={{ r: 3 }} strokeDasharray="5 5" />
            </>
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

const BarParamChart = ({ data, param, title, metric = 'combined' }) => {
  if (!data || data.length < 1) return null;

  return (
    <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 24 }}>
      <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 16, color: '#374151' }}>{title}</h3>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={data} margin={{ top: 10, right: 30, left: 20, bottom: 40 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey={param} stroke="#6b7280" tick={{ fill: '#374151', fontSize: 10, angle: -45, textAnchor: 'end' }} height={60} />
          <YAxis stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
          <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} />
          <Bar dataKey={metric} name={metric === 'combined' ? 'Combined Score' : metric} fill="#2563eb" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

function App() {
  const [files, setFiles] = useState([]);
  const [rawData, setRawData] = useState([]);
  const [selectedModel, setSelectedModel] = useState('all');
  const [selectedSource, setSelectedSource] = useState('all');

  const handleFileUpload = useCallback((e) => {
    Array.from(e.target.files).forEach(file => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const parsed = parseCSV(event.target.result, file.name);
        setFiles(prev => [...prev, file.name]);
        setRawData(prev => [...prev, ...parsed]);
      };
      reader.readAsText(file);
    });
  }, []);

  const removeFile = useCallback((fileName) => {
    setFiles(prev => prev.filter(f => f !== fileName));
    setRawData(prev => prev.filter(d => d._source !== fileName));
  }, []);

  const datasetType = useMemo(() => detectDatasetType(rawData), [rawData]);
  
  const processedData = useMemo(() => {
    if (datasetType === 'truthfulqa') return processDataTruthfulQA(rawData);
    if (datasetType === 'longbench') return processDataLongBench(rawData);
    if (datasetType === 'speed') return processDataSpeed(rawData);
    return [];
  }, [rawData, datasetType]);
  
  const hasBleurtData = useMemo(() => datasetType === 'truthfulqa' && processedData.some(d => d.hasBleurt), [processedData, datasetType]);
  const models = useMemo(() => [...new Set(processedData.map(d => d.model_short))].sort(), [processedData]);

  const modelColors = useMemo(() => {
    const colors = {};
    
    // Simple hash function to generate consistent colors for model names
    const hashColor = (str) => {
      let hash = 0;
      for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
      }
      // Generate color in HSL space for better distribution
      const h = Math.abs(hash % 360);
      const s = 65 + (Math.abs(hash >> 8) % 20); // 65-85% saturation
      const l = 40 + (Math.abs(hash >> 16) % 20); // 40-60% lightness
      return `hsl(${h}, ${s}%, ${l}%)`;
    };
    
    models.forEach((m, i) => { 
      // Use palette for first 30, then hash-based colors
      colors[m] = i < colorPalette.length 
        ? colorPalette[i] 
        : hashColor(m);
    });
    return colors;
  }, [models]);

  const filteredData = useMemo(() => {
    let data = processedData;
    if (selectedModel !== 'all') data = data.filter(d => d.model_short === selectedModel);
    if (selectedSource !== 'all') data = data.filter(d => d._source === selectedSource);
    return data;
  }, [processedData, selectedModel, selectedSource]);

  // Aggregate stats for all parameters
  const threadStats = useMemo(() => aggregateByParam(filteredData, 'threads', datasetType), [filteredData, datasetType]);
  const tempStats = useMemo(() => aggregateByParam(filteredData, 'temperature', datasetType), [filteredData, datasetType]);
  const repPenStats = useMemo(() => aggregateByParam(filteredData, 'repeat_penalty', datasetType), [filteredData, datasetType]);
  const topPStats = useMemo(() => aggregateByParam(filteredData, 'top_p', datasetType), [filteredData, datasetType]);
  const topKStats = useMemo(() => aggregateByParam(filteredData, 'top_k', datasetType), [filteredData, datasetType]);
  const minPStats = useMemo(() => aggregateByParam(filteredData, 'min_p', datasetType), [filteredData, datasetType]);
  const ctxStats = useMemo(() => aggregateByParam(filteredData, 'ctx_size', datasetType), [filteredData, datasetType]);
  const keepStats = useMemo(() => aggregateByParam(filteredData, 'keep', datasetType), [filteredData, datasetType]);
  const batchStats = useMemo(() => aggregateByParam(filteredData, 'batch_size', datasetType), [filteredData, datasetType]);
  const ubatchStats = useMemo(() => aggregateByParam(filteredData, 'ubatch_size', datasetType), [filteredData, datasetType]);
  const threadsStats = useMemo(() => aggregateByParam(filteredData, 'threads', datasetType), [filteredData, datasetType]);
  const nglStats = useMemo(() => aggregateByParam(filteredData, 'ngl', datasetType), [filteredData, datasetType]);
  const flashStats = useMemo(() => aggregateByParam(filteredData, 'flash_attn', datasetType), [filteredData, datasetType]);
  const ctkStats = useMemo(() => aggregateByParam(filteredData, 'ctk', datasetType), [filteredData, datasetType]);
  const ctvStats = useMemo(() => aggregateByParam(filteredData, 'ctv', datasetType), [filteredData, datasetType]);
  const mmapStats = useMemo(() => aggregateByParam(filteredData, 'use_mmap', datasetType), [filteredData, datasetType]);
  const pollStats = useMemo(() => aggregateByParam(filteredData, 'poll_level', datasetType), [filteredData, datasetType]);
  const contextShiftStats = useMemo(() => aggregateByParam(filteredData, 'context_shift', datasetType), [filteredData, datasetType]);
  const splitModeStats = useMemo(() => aggregateByParam(filteredData, 'split_mode', datasetType), [filteredData, datasetType]);
  const dryMultStats = useMemo(() => aggregateByParam(filteredData, 'dry_multiplier', datasetType), [filteredData, datasetType]);
  const freqPenStats = useMemo(() => aggregateByParam(filteredData, 'frequency_penalty', datasetType), [filteredData, datasetType]);
  const tokenLimitStats = useMemo(() => aggregateByParam(filteredData, 'token_limit', datasetType), [filteredData, datasetType]);
  const streamllmStats = useMemo(() => aggregateByParam(filteredData, 'streamllm_enabled', datasetType), [filteredData, datasetType]);
  const sinkCountStats = useMemo(() => aggregateByParam(filteredData, 'sink_count', datasetType), [filteredData, datasetType]);
  const presencePenStats = useMemo(() => aggregateByParam(filteredData, 'presence_penalty', datasetType), [filteredData, datasetType]);

  const modelStats = useMemo(() => {
    if (processedData.length === 0) return [];
    const groups = {};
    const dataToUse = selectedSource === 'all' ? processedData : processedData.filter(d => d._source === selectedSource);
    dataToUse.forEach(d => {
      if (!groups[d.model_short]) {
        groups[d.model_short] = { model: d.model_short, speed: [], combined: [] };
        if (datasetType === 'truthfulqa') {
          groups[d.model_short].accuracy = [];
          groups[d.model_short].bleurt = [];
        } else if (datasetType === 'longbench') {
          groups[d.model_short].rouge1 = [];
          groups[d.model_short].rouge2 = [];
          groups[d.model_short].rougeL = [];
          groups[d.model_short].rougeLsum = [];
        } else if (datasetType === 'speed') {
          groups[d.model_short].prefill_speed = [];
          groups[d.model_short].decode_speed = [];
        }
      }
      if (datasetType === 'speed') {
        groups[d.model_short].prefill_speed.push(d.prefill_speed);
        groups[d.model_short].decode_speed.push(d.decode_speed);
      } else {
        groups[d.model_short].speed.push(d.decode_speed);
      }
      groups[d.model_short].combined.push(d.combined);
      
      if (datasetType === 'truthfulqa') {
        groups[d.model_short].accuracy.push(d.accuracy);
        if (d.bleurt_score !== undefined && !isNaN(d.bleurt_score)) groups[d.model_short].bleurt.push(d.bleurt_score);
      } else if (datasetType === 'longbench') {
        if (d.rouge1) groups[d.model_short].rouge1.push(d.rouge1);
        if (d.rouge2) groups[d.model_short].rouge2.push(d.rouge2);
        if (d.rougeL) groups[d.model_short].rougeL.push(d.rougeL);
        if (d.rougeLsum) groups[d.model_short].rougeLsum.push(d.rougeLsum);
      }
    });
    return Object.values(groups).map(g => {
      const result = {
        model: g.model,
        combined: g.combined.reduce((a,b) => a+b, 0) / g.combined.length,
        count: g.combined.length
      };
      
      if (datasetType === 'speed') {
        result.prefill_speed = g.prefill_speed.reduce((a,b) => a+b, 0) / g.prefill_speed.length;
        result.decode_speed = g.decode_speed.reduce((a,b) => a+b, 0) / g.decode_speed.length;
      } else {
        result.speed = g.speed.reduce((a,b) => a+b, 0) / g.speed.length;
      }
      
      if (datasetType === 'truthfulqa') {
        result.accuracy = g.accuracy.reduce((a,b) => a+b, 0) / g.accuracy.length;
        result.bleurt = g.bleurt.length > 0 ? g.bleurt.reduce((a,b) => a+b, 0) / g.bleurt.length : null;
      } else if (datasetType === 'longbench') {
        result.rouge1 = g.rouge1.length > 0 ? g.rouge1.reduce((a,b) => a+b, 0) / g.rouge1.length : null;
        result.rouge2 = g.rouge2.length > 0 ? g.rouge2.reduce((a,b) => a+b, 0) / g.rouge2.length : null;
        result.rougeL = g.rougeL.length > 0 ? g.rougeL.reduce((a,b) => a+b, 0) / g.rougeL.length : null;
        result.rougeLsum = g.rougeLsum.length > 0 ? g.rougeLsum.reduce((a,b) => a+b, 0) / g.rougeLsum.length : null;
      }
      
      return result;
    }).sort((a,b) => b.combined - a.combined);
  }, [processedData, selectedSource, datasetType]);

  const topRuns = useMemo(() => [...filteredData].sort((a,b) => b.combined - a.combined).slice(0, 6), [filteredData]);

  const stats = useMemo(() => {
    if (filteredData.length === 0) return null;
    
    const result = {
      count: filteredData.length,
      maxSpeed: Math.max(...filteredData.map(d => d.decode_speed)),
      maxCombined: Math.max(...filteredData.map(d => d.combined)),
      avgRuntime: filteredData[0].runtime_seconds ? filteredData.reduce((a,b) => a + (b.runtime_seconds || 0), 0) / filteredData.length : null,
    };
    
    if (datasetType === 'truthfulqa') {
      const withBleurt = filteredData.filter(d => d.bleurt_score !== undefined && !isNaN(d.bleurt_score));
      result.maxAcc = Math.max(...filteredData.map(d => d.accuracy));
      result.maxBleurt = withBleurt.length > 0 ? Math.max(...withBleurt.map(d => d.bleurt_score)) : null;
      result.avgBleurt = withBleurt.length > 0 ? withBleurt.reduce((a,b) => a + b.bleurt_score, 0) / withBleurt.length : null;
      result.bleurtCount = withBleurt.length;
    } else if (datasetType === 'longbench') {
      result.maxRougeL = Math.max(...filteredData.map(d => d.rougeL));
      result.maxRouge1 = Math.max(...filteredData.map(d => d.rouge1 || 0));
      result.maxRouge2 = Math.max(...filteredData.map(d => d.rouge2 || 0));
      result.avgRougeL = filteredData.reduce((a,b) => a + b.rougeL, 0) / filteredData.length;
    }
    
    return result;
  }, [filteredData, datasetType]);

  // Empty state
  if (rawData.length === 0) {
    return (
      <div style={{ minHeight: '100vh', background: '#ffffff', color: '#111827', fontFamily: "'Inter', 'Helvetica', 'Arial', sans-serif", display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '32px' }}>
        <h1 style={{ fontSize: '2rem', fontWeight: 700, color: '#111827', marginBottom: 16, letterSpacing: '-0.025em' }}>
          LLM Hyperparameter Analyzer
        </h1>
        <p style={{ color: '#6b7280', marginBottom: 32 }}>Upload CSV files from your hyperparameter search</p>

        <label style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '48px 64px', border: '2px dashed #d1d5db', borderRadius: '12px', cursor: 'pointer', background: '#f9fafb' }}
          onMouseOver={(e) => e.currentTarget.style.borderColor = '#2563eb'}
          onMouseOut={(e) => e.currentTarget.style.borderColor = '#d1d5db'}>
          <svg width="48" height="48" fill="none" stroke="#6b7280" viewBox="0 0 24 24" style={{ marginBottom: 16 }}>
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <span style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 8, color: '#111827' }}>Drop CSV files here</span>
          <span style={{ color: '#6b7280', fontSize: '0.9rem' }}>or click to browse</span>
          <input type="file" accept=".csv" multiple onChange={handleFileUpload} style={{ display: 'none' }} />
        </label>

        <div style={{ color: '#6b7280', marginTop: 24, fontSize: '0.8rem', textAlign: 'center', maxWidth: 700, lineHeight: 1.6 }}>
          <p style={{ marginBottom: 12, fontWeight: 600, color: '#111827' }}>Supported Datasets:</p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, textAlign: 'left' }}>
            <div>
              <p style={{ color: '#059669', fontWeight: 600, marginBottom: 4 }}>✓ TruthfulQA</p>
              <p style={{ fontSize: '0.75rem', color: '#6b7280' }}>accuracy, bleurt_score, avg_decode_speed, ...</p>
            </div>
            <div>
              <p style={{ color: '#7c3aed', fontWeight: 600, marginBottom: 4 }}>✓ LongBench</p>
              <p style={{ fontSize: '0.75rem', color: '#6b7280' }}>rouge1, rouge2, rougeL, rougeLsum, avg_total_speed_tok_s, streamllm_enabled, ...</p>
            </div>
            <div>
              <p style={{ color: '#059669', fontWeight: 600, marginBottom: 4 }}>✓ Speed Only</p>
              <p style={{ fontSize: '0.75rem', color: '#6b7280' }}>model, prefill_speed/prefill_tps, decode_speed/decoding_tps</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const CustomTooltip = datasetType === 'truthfulqa' ? CustomTooltipTruthfulQA : CustomTooltipLongBench;
  const datasetLabel = datasetType === 'truthfulqa' ? 'TruthfulQA' : datasetType === 'longbench' ? 'LongBench' : datasetType === 'speed' ? 'Speed Comparison' : 'Unknown';
  const datasetColor = datasetType === 'truthfulqa' ? '#10b981' : datasetType === 'speed' ? '#10b981' : '#8b5cf6';

  return (
    <div style={{ minHeight: '100vh', background: '#ffffff', color: '#1f2937', fontFamily: "'Inter', 'Helvetica', 'Arial', sans-serif", padding: '24px' }}>
      {/* Header */}
      <div style={{ marginBottom: 24, textAlign: 'center', borderBottom: '2px solid #e5e7eb', paddingBottom: 16 }}>
        <h1 style={{ fontSize: '1.75rem', fontWeight: 700, color: '#111827', marginBottom: 8, letterSpacing: '-0.025em' }}>
          LLM Hyperparameter Analysis
        </h1>
        <p style={{ color: '#6b7280', fontSize: '0.9rem' }}>
          {filteredData.length} runs • {files.length} file(s) • <span style={{ color: '#374151', fontWeight: 600 }}>{datasetLabel}</span>
          {hasBleurtData && ' • BLEURT included'}
        </p>
      </div>

      {/* File Management */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20, alignItems: 'center' }}>
        <span style={{ color: '#374151', fontSize: '0.8rem' }}>Files:</span>
        {files.map(f => (
          <span key={f} style={{ background: 'rgba(59, 130, 246, 0.2)', border: '1px solid rgba(59, 130, 246, 0.4)', borderRadius: 6, padding: '3px 8px', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: 6 }}>
            {f}
            <button onClick={() => removeFile(f)} style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', padding: 0 }}>×</button>
          </span>
        ))}
        <label style={{ background: 'rgba(16, 185, 129, 0.2)', border: '1px solid rgba(16, 185, 129, 0.4)', borderRadius: 6, padding: '3px 10px', fontSize: '0.8rem', cursor: 'pointer' }}>
          + Add <input type="file" accept=".csv" multiple onChange={handleFileUpload} style={{ display: 'none' }} />
        </label>
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <label style={{ color: '#374151', fontSize: '0.8rem', marginRight: 6 }}>Model:</label>
          <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 6, padding: '6px 10px', color: '#111827', cursor: 'pointer', fontSize: '0.85rem' }}>
            <option value="all">All Models</option>
            {models.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
        {files.length > 1 && (
          <div>
            <label style={{ color: '#374151', fontSize: '0.8rem', marginRight: 6 }}>File:</label>
            <select value={selectedSource} onChange={e => setSelectedSource(e.target.value)} style={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 6, padding: '6px 10px', color: '#111827', cursor: 'pointer', fontSize: '0.85rem' }}>
              <option value="all">All Files</option>
              {files.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>
        )}
      </div>

      {/* Top Runs */}
      <div style={{ marginBottom: 32 }}>
        <h2 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: 12, color: '#111827' }}>
          {datasetType === 'speed' ? '⚡ Top Speed Performers' : `🏆 Top Configurations ${datasetType === 'truthfulqa' && hasBleurtData ? '(Speed + Accuracy + BLEURT)' : datasetType === 'truthfulqa' ? '(Speed + Accuracy)' : '(Speed + ROUGE scores)'}`}
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 12 }}>
          {topRuns.map((run, i) => (
            <div key={`${run._source}-${run.run_id || i}`} style={{
              background: i === 0 ? 'linear-gradient(135deg, rgba(251, 191, 36, 0.15), rgba(245, 158, 11, 0.05))' : 'rgba(30, 41, 59, 0.6)',
              border: i === 0 ? '2px solid rgba(251, 191, 36, 0.4)' : '1px solid rgba(148, 163, 184, 0.15)',
              borderRadius: 10, padding: 16, position: 'relative',
            }}>
              <div style={{ position: 'absolute', top: 10, right: 10, background: i === 0 ? '#fbbf24' : '#475569', color: i === 0 ? '#0f172a' : '#e2e8f0', borderRadius: 12, padding: '2px 10px', fontSize: '0.7rem', fontWeight: 700 }}>#{i + 1}</div>
              <div style={{ fontSize: '0.95rem', fontWeight: 700, color: modelColors[run.model_short], marginBottom: 8 }}>{run.model_short}</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '4px 8px', fontSize: '0.75rem' }}>
                {datasetType === 'speed' ? (
                  <>
                    <div><span style={{ color: '#6b7280' }}>Combined: </span><span style={{ fontWeight: 700, color: '#fbbf24' }}>{run.combined?.toFixed(3)}</span></div>
                    <div><span style={{ color: '#6b7280' }}>Prefill: </span><span style={{ fontWeight: 600, color: '#059669' }}>{run.prefill_speed?.toFixed(1)} tok/s</span></div>
                    <div><span style={{ color: '#6b7280' }}>Decode: </span><span style={{ fontWeight: 600, color: '#8b5cf6' }}>{run.decode_speed?.toFixed(1)} tok/s</span></div>
                  </>
                ) : (
                  <>
                    <div><span style={{ color: '#6b7280' }}>Score: </span><span style={{ fontWeight: 700, color: '#fbbf24' }}>{run.combined?.toFixed(3)}</span></div>
                    
                    {datasetType === 'truthfulqa' && (
                      <>
                        <div><span style={{ color: '#6b7280' }}>Acc: </span><span style={{ fontWeight: 600 }}>{(run.accuracy * 100).toFixed(1)}%</span></div>
                        {run.bleurt_score !== undefined && !isNaN(run.bleurt_score) && <div><span style={{ color: '#6b7280' }}>BLEURT: </span><span style={{ color: '#06b6d4' }}>{run.bleurt_score?.toFixed(3)}</span></div>}
                      </>
                    )}
                    
                    {datasetType === 'longbench' && (
                      <>
                        <div><span style={{ color: '#6b7280' }}>R-L: </span><span style={{ fontWeight: 600, color: '#8b5cf6' }}>{run.rougeL?.toFixed(4)}</span></div>
                        <div><span style={{ color: '#6b7280' }}>R-1: </span><span>{run.rouge1?.toFixed(4)}</span></div>
                        {run.token_limit !== undefined && <div><span style={{ color: '#6b7280' }}>Tokens: </span>{run.token_limit}</div>}
                        {run.streamllm_enabled !== undefined && <div><span style={{ color: '#6b7280' }}>StreamLLM: </span>{run.streamllm_enabled ? '✓' : '✗'}</div>}
                        {run.sink_count !== undefined && run.streamllm_enabled && <div><span style={{ color: '#6b7280' }}>Sinks: </span>{run.sink_count}</div>}
                      </>
                    )}
                    
                    <div><span style={{ color: '#6b7280' }}>Speed: </span><span style={{ fontWeight: 600 }}>{run.decode_speed?.toFixed(1)}</span></div>
                    {run.temperature !== undefined && <div><span style={{ color: '#6b7280' }}>Temp: </span>{run.temperature}</div>}
                    {run.min_p !== undefined && <div><span style={{ color: '#6b7280' }}>Min-p: </span>{run.min_p}</div>}
                    {run.ctx_size !== undefined && <div><span style={{ color: '#6b7280' }}>Ctx: </span>{run.ctx_size}</div>}
                    {run.batch_size !== undefined && <div><span style={{ color: '#6b7280' }}>Batch: </span>{run.batch_size}</div>}
                    {run.presence_penalty !== undefined && <div><span style={{ color: '#6b7280' }}>Pres: </span>{run.presence_penalty}</div>}
                    {run.runtime_seconds !== undefined && <div><span style={{ color: '#6b7280' }}>Time: </span>{run.runtime_seconds}s</div>}
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Speed Dataset Visualizations */}
      {datasetType === 'speed' && (
        <>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 16, color: '#111827', borderBottom: '2px solid #e5e7eb', paddingBottom: 8, marginTop: 32 }}>⚡ Speed Analysis</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: 20, marginBottom: 24 }}>
            {/* Prefill vs Decode Speed Scatter */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Prefill vs Decode Speed</h3>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" dataKey="prefill_speed" name="Prefill" unit=" tok/s" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <YAxis type="number" dataKey="decode_speed" name="Decode" unit=" tok/s" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8, fontSize: 12 }} formatter={(value, name) => [value?.toFixed(1) + ' tok/s', name]} />
                  <Scatter data={filteredData} fill="#059669">
                    {filteredData.map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model_short] || '#10b981'} fillOpacity={0.8} />)}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
                      <ScatterLegend modelColors={modelColors} models={models} />
            </div>

            {/* Model Average Speeds */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Average Speeds by Model</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={modelStats} layout="vertical" margin={{ top: 10, right: 30, left: 100, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <YAxis type="category" dataKey="model" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 10 }} width={95} />
                  <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={v => v?.toFixed(1) + ' tok/s'} />
                  <Legend />
                  <Bar dataKey="prefill_speed" name="Prefill" fill="#059669" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="decode_speed" name="Decode" fill="#7c3aed" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Speed Distribution Box Plot Style */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Decode Speed Distribution</h3>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 60, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="category" dataKey="model_short" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 9, angle: -45, textAnchor: 'end' }} height={80} />
                  <YAxis type="number" dataKey="decode_speed" name="Decode Speed" unit=" tok/s" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={(value) => value?.toFixed(1) + ' tok/s'} />
                  <Scatter data={filteredData} fill="#7c3aed">
                    {filteredData.map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model_short] || '#8b5cf6'} fillOpacity={0.6} />)}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
                      <ScatterLegend modelColors={modelColors} models={models} />
            </div>

            {/* Prefill Speed Distribution */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Prefill Speed Distribution</h3>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 60, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="category" dataKey="model_short" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 9, angle: -45, textAnchor: 'end' }} height={80} />
                  <YAxis type="number" dataKey="prefill_speed" name="Prefill Speed" unit=" tok/s" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={(value) => value?.toFixed(1) + ' tok/s'} />
                  <Scatter data={filteredData} fill="#059669">
                    {filteredData.map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model_short] || '#10b981'} fillOpacity={0.6} />)}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
                      <ScatterLegend modelColors={modelColors} models={models} />
            </div>

            {/* Model Speed Rankings */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Combined Speed Score</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={modelStats} layout="vertical" margin={{ top: 10, right: 30, left: 100, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" domain={[0, 1]} stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <YAxis type="category" dataKey="model" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 10 }} width={95} />
                  <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={v => v?.toFixed(3)} />
                  <Bar dataKey="combined" name="Combined Score" radius={[0, 4, 4, 0]}>
                    {modelStats.map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model] || '#fbbf24'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Speed Ratio Analysis */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Prefill/Decode Speed Ratio</h3>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" dataKey="decode_speed" name="Decode Speed" unit=" tok/s" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <YAxis type="number" dataKey="prefill_speed" name="Prefill Speed" unit=" tok/s" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={(value) => value?.toFixed(1) + ' tok/s'} />
                  <Scatter data={filteredData} fill="#06b6d4">
                    {filteredData.map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model_short] || '#06b6d4'} fillOpacity={0.7} />)}
                  </Scatter>
                  {/* Reference line for 1:1 ratio */}
                  <line x1={0} y1={0} x2={100} y2={100} stroke="#94a3b8" strokeWidth={1} strokeDasharray="5 5" opacity={0.3} />
                </ScatterChart>
              </ResponsiveContainer>
                      <ScatterLegend modelColors={modelColors} models={models} />
            </div>
          </div>

          {/* Speed Statistics Summary */}
          <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 16, padding: 20, marginBottom: 24 }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 12, color: '#059669' }}>📊 Speed Statistics</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 16 }}>
              {modelStats.slice(0, 4).map((stat, idx) => (
                <div key={idx} style={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8, padding: 12 }}>
                  <div style={{ color: modelColors[stat.model] || '#10b981', fontSize: '0.9rem', fontWeight: 600, marginBottom: 8 }}>
                    {stat.model}
                  </div>
                  <div style={{ fontSize: '0.75rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 8px' }}>
                    <span style={{ color: '#374151' }}>Prefill:</span><span style={{ fontWeight: 600, color: '#059669' }}>{stat.prefill_speed?.toFixed(1)} tok/s</span>
                    <span style={{ color: '#374151' }}>Decode:</span><span style={{ fontWeight: 600, color: '#8b5cf6' }}>{stat.decode_speed?.toFixed(1)} tok/s</span>
                    <span style={{ color: '#374151' }}>Combined:</span><span style={{ fontWeight: 600 }}>{stat.combined?.toFixed(3)}</span>
                    <span style={{ color: '#374151' }}>Samples:</span><span>{stat.count}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {/* Main Scatter/Bar Charts (Skip for speed-only dataset) */}
      {datasetType !== 'speed' && (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: 20, marginBottom: 24 }}>
        {/* Speed vs Quality Metric */}
        <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>
            {datasetType === 'truthfulqa' ? 'Speed vs Accuracy' : 'Speed vs ROUGE-L'}
          </h3>
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis type="number" dataKey="decode_speed" name="Speed" unit=" tok/s" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
              <YAxis type="number" dataKey={datasetType === 'truthfulqa' ? 'accuracy' : 'rougeL'} 
                name={datasetType === 'truthfulqa' ? 'Accuracy' : 'ROUGE-L'} 
                stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} 
                tickFormatter={v => datasetType === 'truthfulqa' ? `${(v*100).toFixed(0)}%` : v.toFixed(3)} />
              <Tooltip content={<CustomTooltip modelColors={modelColors} />} />
              <Scatter data={filteredData} fill="#2563eb">
                {filteredData.map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model_short] || '#3b82f6'} fillOpacity={0.8} />)}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <ScatterLegend modelColors={modelColors} models={models} />
        </div>

        {/* ROUGE-1 vs ROUGE-L (LongBench only) */}
        {datasetType === 'longbench' && (
          <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>ROUGE-1 vs ROUGE-L Correlation</h3>
            <ResponsiveContainer width="100%" height={320}>
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis type="number" dataKey="rougeL" name="ROUGE-L" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                <YAxis type="number" dataKey="rouge1" name="ROUGE-1" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                <Tooltip content={<CustomTooltip modelColors={modelColors} />} />
                <Scatter data={filteredData} fill="#ec4899">
                  {filteredData.map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model_short] || '#ec4899'} fillOpacity={0.8} />)}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
            <ScatterLegend modelColors={modelColors} models={models} />
          </div>
        )}

        {/* Model Combined Score */}
        <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Model Combined Score</h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={modelStats} layout="vertical" margin={{ top: 10, right: 30, left: 80, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis type="number" domain={[0, 1]} stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
              <YAxis type="category" dataKey="model" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 10 }} width={75} />
              <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} />
              <Bar dataKey="combined" name="Combined" radius={[0, 4, 4, 0]}>
                {modelStats.map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model] || '#3b82f6'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Model ROUGE-L Scores (LongBench only) */}
        {datasetType === 'longbench' && (
          <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Model ROUGE-L Scores</h3>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={modelStats} layout="vertical" margin={{ top: 10, right: 30, left: 80, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis type="number" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                <YAxis type="category" dataKey="model" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 10 }} width={75} />
                <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={v => v?.toFixed(4)} />
                <Bar dataKey="rougeL" name="ROUGE-L" fill="#7c3aed" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
      )}

      {/* TruthfulQA: Detailed Accuracy & BLEURT Analysis Section */}
      {datasetType === 'truthfulqa' && (
        <>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 16, color: '#111827', borderBottom: '2px solid #e5e7eb', paddingBottom: 8, marginTop: 32 }}>📊 Detailed Accuracy & BLEURT Analysis</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: 20, marginBottom: 24 }}>
            
            {/* Accuracy Distribution by Model (Box Plot Style) */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Accuracy Distribution by Model</h3>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 80, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="category" dataKey="model_short" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 9, angle: -45, textAnchor: 'end' }} height={80} />
                  <YAxis type="number" dataKey="accuracy" name="Accuracy" domain={[0, 1]} stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={(value) => `${(value*100).toFixed(1)}%`} />
                  <Scatter data={filteredData} fill="#059669">
                    {filteredData.map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model_short] || '#10b981'} fillOpacity={0.7} />)}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
                      <ScatterLegend modelColors={modelColors} models={models} />
            </div>

            {/* BLEURT Distribution by Model */}
            {hasBleurtData && (
              <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
                <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>BLEURT Distribution by Model</h3>
                <ResponsiveContainer width="100%" height={320}>
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 80, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis type="category" dataKey="model_short" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 9, angle: -45, textAnchor: 'end' }} height={80} />
                    <YAxis type="number" dataKey="bleurt_score" name="BLEURT" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={(value) => value?.toFixed(3)} />
                    <Scatter data={filteredData.filter(d => d.bleurt_score !== null && d.bleurt_score !== undefined)} fill="#06b6d4">
                      {filteredData.filter(d => d.bleurt_score !== null && d.bleurt_score !== undefined).map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model_short] || '#06b6d4'} fillOpacity={0.7} />)}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
                        <ScatterLegend modelColors={modelColors} models={models} />
              </div>
            )}

            {/* Accuracy vs BLEURT Correlation */}
            {hasBleurtData && (
              <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
                <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Accuracy vs BLEURT Correlation</h3>
                <ResponsiveContainer width="100%" height={320}>
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis type="number" dataKey="accuracy" name="Accuracy" domain={[0, 1]} stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
                    <YAxis type="number" dataKey="bleurt_score" name="BLEURT" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                    <Tooltip content={<CustomTooltip modelColors={modelColors} />} />
                    <Scatter data={filteredData.filter(d => d.bleurt_score !== null && d.bleurt_score !== undefined)} fill="#7c3aed">
                      {filteredData.filter(d => d.bleurt_score !== null && d.bleurt_score !== undefined).map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model_short] || '#8b5cf6'} fillOpacity={0.7} />)}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
                        <ScatterLegend modelColors={modelColors} models={models} />
              </div>
            )}

            {/* Model Accuracy Rankings */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Model Accuracy Rankings</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={modelStats} layout="vertical" margin={{ top: 10, right: 30, left: 120, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" domain={[0, 1]} stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
                  <YAxis type="category" dataKey="model" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 10 }} width={115} />
                  <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={v => `${(v*100).toFixed(1)}%`} />
                  <Bar dataKey="accuracy" name="Accuracy" radius={[0, 4, 4, 0]}>
                    {modelStats.map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model] || '#10b981'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* BLEURT Rankings */}
            {hasBleurtData && (
              <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
                <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Model BLEURT Rankings</h3>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={modelStats.filter(m => m.bleurt !== null)} layout="vertical" margin={{ top: 10, right: 30, left: 120, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis type="number" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                    <YAxis type="category" dataKey="model" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 10 }} width={115} />
                    <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={v => v?.toFixed(3)} />
                    <Bar dataKey="bleurt" name="BLEURT" fill="#06b6d4" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Accuracy Breakdown (High/Medium/Low) */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Accuracy Breakdown</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={(() => {
                  const bins = [
                    { range: '0-40%', count: 0, color: '#ef4444' },
                    { range: '40-60%', count: 0, color: '#f59e0b' },
                    { range: '60-80%', count: 0, color: '#059669' },
                    { range: '80-100%', count: 0, color: '#06b6d4' }
                  ];
                  filteredData.forEach(d => {
                    const acc = d.accuracy * 100;
                    if (acc < 40) bins[0].count++;
                    else if (acc < 60) bins[1].count++;
                    else if (acc < 80) bins[2].count++;
                    else bins[3].count++;
                  });
                  return bins;
                })()} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="range" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <YAxis stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} />
                  <Bar dataKey="count" name="Number of Runs" radius={[4, 4, 0, 0]}>
                    {[0, 1, 2, 3].map((index) => <Cell key={`cell-${index}`} fill={['#ef4444', '#f59e0b', '#10b981', '#06b6d4'][index]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Speed vs Accuracy vs BLEURT (Bubble Chart) */}
            {hasBleurtData && (
              <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
                <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Speed vs Accuracy (Size = BLEURT)</h3>
                <ResponsiveContainer width="100%" height={320}>
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis type="number" dataKey="decode_speed" name="Speed" unit=" tok/s" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                    <YAxis type="number" dataKey="accuracy" name="Accuracy" domain={[0, 1]} stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
                    <Tooltip content={<CustomTooltip modelColors={modelColors} />} />
                    <Scatter data={filteredData.filter(d => d.bleurt_score !== null && d.bleurt_score !== undefined)} fill="#7c3aed">
                      {filteredData.filter(d => d.bleurt_score !== null && d.bleurt_score !== undefined).map((entry, index) => {
                        const bleurtNorm = Math.abs(entry.bleurt_score);
                        const radius = Math.max(3, Math.min(15, bleurtNorm * 20));
                        return <Cell key={`cell-${index}`} fill={modelColors[entry.model_short] || '#8b5cf6'} fillOpacity={0.6} r={radius} />;
                      })}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
                        <ScatterLegend modelColors={modelColors} models={models} />
              </div>
            )}

            {/* Accuracy + BLEURT Combined Metric */}
            {hasBleurtData && (
              <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
                <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Accuracy + BLEURT Combined</h3>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={modelStats.filter(m => m.bleurt !== null).map(m => ({
                    ...m,
                    bleurt_abs: Math.abs(m.bleurt)
                  }))} layout="vertical" margin={{ top: 10, right: 30, left: 120, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis type="number" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                    <YAxis type="category" dataKey="model" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 10 }} width={115} />
                    <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} />
                    <Legend />
                    <Bar dataKey="accuracy" name="Accuracy" fill="#059669" radius={[0, 0, 0, 0]} stackId="a" />
                    <Bar dataKey="bleurt_abs" name="BLEURT (abs)" fill="#06b6d4" radius={[0, 4, 4, 0]} stackId="a" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Accuracy Consistency (Std Dev by Model) */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Accuracy Consistency (Lower = More Consistent)</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={(() => {
                  const groups = {};
                  filteredData.forEach(d => {
                    if (!groups[d.model_short]) groups[d.model_short] = [];
                    groups[d.model_short].push(d.accuracy);
                  });
                  return Object.entries(groups).map(([model, accuracies]) => {
                    const mean = accuracies.reduce((a, b) => a + b, 0) / accuracies.length;
                    const variance = accuracies.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / accuracies.length;
                    const stdDev = Math.sqrt(variance);
                    return { model, stdDev, count: accuracies.length };
                  }).sort((a, b) => a.stdDev - b.stdDev);
                })()} layout="vertical" margin={{ top: 10, right: 30, left: 120, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <YAxis type="category" dataKey="model" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 10 }} width={115} />
                  <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={v => v?.toFixed(4)} />
                  <Bar dataKey="stdDev" name="Std Dev" radius={[0, 4, 4, 0]}>
                    {(() => {
                      const groups = {};
                      filteredData.forEach(d => {
                        if (!groups[d.model_short]) groups[d.model_short] = [];
                        groups[d.model_short].push(d.accuracy);
                      });
                      return Object.entries(groups).map(([model, accuracies]) => {
                        const mean = accuracies.reduce((a, b) => a + b, 0) / accuracies.length;
                        const variance = accuracies.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / accuracies.length;
                        const stdDev = Math.sqrt(variance);
                        return { model, stdDev };
                      }).sort((a, b) => a.stdDev - b.stdDev);
                    })().map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model] || '#8b5cf6'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

          </div>

          {/* TruthfulQA Summary Statistics */}
          <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 16, padding: 20, marginBottom: 24 }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 12, color: '#059669' }}>📈 TruthfulQA Summary Statistics</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
              {modelStats.slice(0, 4).map((stat, idx) => (
                <div key={idx} style={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8, padding: 12 }}>
                  <div style={{ color: modelColors[stat.model] || '#10b981', fontSize: '0.9rem', fontWeight: 600, marginBottom: 8 }}>
                    {stat.model}
                  </div>
                  <div style={{ fontSize: '0.75rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 8px' }}>
                    <span style={{ color: '#374151' }}>Accuracy:</span><span style={{ fontWeight: 600, color: '#059669' }}>{(stat.accuracy * 100).toFixed(1)}%</span>
                    {stat.bleurt !== null && <><span style={{ color: '#374151' }}>BLEURT:</span><span style={{ fontWeight: 600, color: '#06b6d4' }}>{stat.bleurt?.toFixed(3)}</span></>}
                    <span style={{ color: '#374151' }}>Speed:</span><span style={{ fontWeight: 600 }}>{stat.speed?.toFixed(1)} tok/s</span>
                    <span style={{ color: '#374151' }}>Samples:</span><span>{stat.count}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {/* LongBench: Detailed ROUGE Analysis Section */}
      {datasetType === 'longbench' && (
        <>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 16, color: '#111827', borderBottom: '2px solid #e5e7eb', paddingBottom: 8, marginTop: 32 }}>📊 Detailed ROUGE Score Analysis</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: 20, marginBottom: 24 }}>
            {/* ROUGE Score Comparison by Model */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>All ROUGE Metrics by Model</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={modelStats} margin={{ top: 10, right: 30, left: 80, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <YAxis type="category" dataKey="model" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 10 }} width={75} />
                  <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={v => v?.toFixed(4)} />
                  <Legend />
                  <Bar dataKey="rouge1" name="ROUGE-1" fill="#059669" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="rouge2" name="ROUGE-2" fill="#f59e0b" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="rougeL" name="ROUGE-L" fill="#7c3aed" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="rougeLsum" name="ROUGE-Lsum" fill="#06b6d4" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* ROUGE-2 vs ROUGE-L Scatter */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>ROUGE-2 vs ROUGE-L (Bigram Overlap)</h3>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" dataKey="rougeL" name="ROUGE-L" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <YAxis type="number" dataKey="rouge2" name="ROUGE-2" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip modelColors={modelColors} />} />
                  <Scatter data={filteredData} fill="#f59e0b">
                    {filteredData.map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model_short] || '#f59e0b'} fillOpacity={0.8} />)}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
                      <ScatterLegend modelColors={modelColors} models={models} />
            </div>

            {/* ROUGE-Lsum vs ROUGE-L */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>ROUGE-Lsum vs ROUGE-L</h3>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" dataKey="rougeL" name="ROUGE-L" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <YAxis type="number" dataKey="rougeLsum" name="ROUGE-Lsum" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip modelColors={modelColors} />} />
                  <Scatter data={filteredData.filter(d => d.rougeLsum)} fill="#06b6d4">
                    {filteredData.filter(d => d.rougeLsum).map((entry, index) => <Cell key={`cell-${index}`} fill={modelColors[entry.model_short] || '#06b6d4'} fillOpacity={0.8} />)}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
                      <ScatterLegend modelColors={modelColors} models={models} />
            </div>

            {/* ROUGE Score Distribution */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>ROUGE Score Distribution</h3>
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={[...filteredData].sort((a, b) => a.rougeL - b.rougeL).map((d, i) => ({ ...d, index: i }))} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="index" name="Run Index" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <YAxis stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip modelColors={modelColors} />} />
                  <Legend />
                  <Line type="monotone" dataKey="rouge1" name="ROUGE-1" stroke="#10b981" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="rouge2" name="ROUGE-2" stroke="#f59e0b" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="rougeL" name="ROUGE-L" stroke="#8b5cf6" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="rougeLsum" name="ROUGE-Lsum" stroke="#06b6d4" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {/* Keep Parameter Analysis Section */}
      {keepStats.length >= 2 && (
        <>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 16, color: '#111827', borderBottom: '2px solid #e5e7eb', paddingBottom: 8, marginTop: 32 }}>🔑 Keep Parameter Analysis (--keep 0 vs --keep 4)</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: 20, marginBottom: 24 }}>
            {/* Keep Impact on ROUGE-L */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Keep Impact on ROUGE-L</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={keepStats} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="keep" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} label={{ value: '--keep value', position: 'insideBottom', offset: -10, fill: '#374151' }} />
                  <YAxis stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={v => v?.toFixed(4)} />
                  <Legend />
                  {datasetType === 'longbench' && (
                    <>
                      <Bar dataKey="rougeL" name="ROUGE-L" fill="#7c3aed" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="rouge1" name="ROUGE-1" fill="#059669" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="rouge2" name="ROUGE-2" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                    </>
                  )}
                  {datasetType === 'truthfulqa' && (
                    <Bar dataKey="accuracy" name="Accuracy" fill="#7c3aed" radius={[4, 4, 0, 0]} />
                  )}
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Keep Impact on Speed */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Keep Impact on Speed</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={keepStats} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="keep" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} label={{ value: '--keep value', position: 'insideBottom', offset: -10, fill: '#374151' }} />
                  <YAxis stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={v => v?.toFixed(1)} />
                  <Bar dataKey="speed" name="Speed (tok/s)" fill="#059669" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Keep: Speed vs ROUGE-L Scatter */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Speed vs ROUGE-L (Keep Color-Coded)</h3>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" dataKey="decode_speed" name="Speed" unit=" tok/s" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <YAxis type="number" dataKey={datasetType === 'longbench' ? 'rougeL' : 'accuracy'} name={datasetType === 'longbench' ? 'ROUGE-L' : 'Accuracy'} stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip modelColors={modelColors} />} />
                  <Legend />
                  <Scatter name="--keep 0" data={filteredData.filter(d => d.keep === 0 || d.keep === '0')} fill="#f59e0b">
                    {filteredData.filter(d => d.keep === 0 || d.keep === '0').map((entry, index) => <Cell key={`keep0-${index}`} fill="#f59e0b" fillOpacity={0.8} />)}
                  </Scatter>
                  <Scatter name="--keep 4" data={filteredData.filter(d => d.keep === 4 || d.keep === '4')} fill="#06b6d4">
                    {filteredData.filter(d => d.keep === 4 || d.keep === '4').map((entry, index) => <Cell key={`keep4-${index}`} fill="#06b6d4" fillOpacity={0.8} />)}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
                      <ScatterLegend modelColors={modelColors} models={models} />
            </div>

            {/* Keep Combined Score Comparison */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Combined Score: Keep 0 vs Keep 4</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={keepStats} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="keep" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} label={{ value: '--keep value', position: 'insideBottom', offset: -10, fill: '#374151' }} />
                  <YAxis stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={v => v?.toFixed(3)} />
                  <Bar dataKey="combined" name="Combined Score" fill="#fbbf24" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Keep Runtime Comparison */}
            {keepStats.some(s => s.runtime !== null) && (
              <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
                <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Runtime Comparison</h3>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={keepStats.filter(s => s.runtime !== null)} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="keep" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} label={{ value: '--keep value', position: 'insideBottom', offset: -10, fill: '#374151' }} />
                    <YAxis stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={v => `${v?.toFixed(1)}s`} />
                    <Bar dataKey="runtime" name="Runtime (seconds)" fill="#ec4899" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Keep Parameter Line Chart */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Keep Parameter Trend</h3>
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={keepStats} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="keep" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <YAxis yAxisId="left" stroke="#10b981" tick={{ fill: '#10b981', fontSize: 11 }} />
                  <YAxis yAxisId="right" orientation="right" stroke="#8b5cf6" tick={{ fill: '#8b5cf6', fontSize: 11 }} domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="speed" name="Speed (tok/s)" stroke="#10b981" strokeWidth={2} dot={{ r: 5 }} />
                  {datasetType === 'longbench' && (
                    <>
                      <Line yAxisId="right" type="monotone" dataKey="rougeL" name="ROUGE-L" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 5 }} />
                      <Line yAxisId="right" type="monotone" dataKey="rouge1" name="ROUGE-1" stroke="#ec4899" strokeWidth={1.5} dot={{ r: 4 }} strokeDasharray="5 5" />
                    </>
                  )}
                  {datasetType === 'truthfulqa' && (
                    <Line yAxisId="right" type="monotone" dataKey="accuracy" name="Accuracy" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 5 }} />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Keep Summary Stats */}
          <div style={{ background: '#fef3c7', border: '1px solid #fde68a', borderRadius: 16, padding: 20, marginBottom: 24 }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 12, color: '#f59e0b' }}>🔍 Keep Parameter Impact Summary</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
              {keepStats.map((stat, idx) => (
                <div key={idx} style={{ background: stat.keep === 0 || stat.keep === '0' ? 'rgba(245, 158, 11, 0.15)' : 'rgba(6, 182, 212, 0.15)', border: `1px solid ${stat.keep === 0 || stat.keep === '0' ? 'rgba(245, 158, 11, 0.4)' : 'rgba(6, 182, 212, 0.4)'}`, borderRadius: 8, padding: 12 }}>
                  <div style={{ color: stat.keep === 0 || stat.keep === '0' ? '#f59e0b' : '#06b6d4', fontSize: '0.9rem', fontWeight: 600, marginBottom: 8 }}>
                    --keep {stat.keep}
                  </div>
                  <div style={{ fontSize: '0.75rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 8px' }}>
                    {datasetType === 'longbench' && (
                      <>
                        <span style={{ color: '#374151' }}>ROUGE-L:</span><span style={{ fontWeight: 600 }}>{stat.rougeL?.toFixed(4) || 'N/A'}</span>
                        <span style={{ color: '#374151' }}>ROUGE-1:</span><span>{stat.rouge1?.toFixed(4) || 'N/A'}</span>
                      </>
                    )}
                    {datasetType === 'truthfulqa' && (
                      <><span style={{ color: '#374151' }}>Accuracy:</span><span style={{ fontWeight: 600 }}>{stat.accuracy?.toFixed(4) || 'N/A'}</span></>
                    )}
                    <span style={{ color: '#374151' }}>Speed:</span><span style={{ fontWeight: 600 }}>{stat.speed?.toFixed(1) || 'N/A'} tok/s</span>
                    <span style={{ color: '#374151' }}>Combined:</span><span style={{ fontWeight: 600 }}>{stat.combined?.toFixed(3) || 'N/A'}</span>
                    <span style={{ color: '#374151' }}>Runs:</span><span>{stat.count}</span>
                    {stat.runtime && <><span style={{ color: '#374151' }}>Runtime:</span><span>{stat.runtime?.toFixed(1)}s</span></>}
                  </div>
                </div>
              ))}
            </div>
            {keepStats.length === 2 && (
              <div style={{ marginTop: 16, padding: '12px', background: 'rgba(251, 191, 36, 0.1)', borderRadius: 8, border: '1px solid rgba(251, 191, 36, 0.3)', fontSize: '0.85rem' }}>
                <span style={{ color: '#fbbf24', fontWeight: 600 }}>💡 Verdict:</span>
                <span style={{ color: '#111827', marginLeft: 8 }}>
                  {(() => {
                    const keep0 = keepStats.find(s => s.keep === 0 || s.keep === '0');
                    const keep4 = keepStats.find(s => s.keep === 4 || s.keep === '4');
                    if (!keep0 || !keep4) return 'Not enough data to compare';
                    
                    const qualityMetric = datasetType === 'longbench' ? 'rougeL' : 'accuracy';
                    const qualityDiff = ((keep4[qualityMetric] - keep0[qualityMetric]) / keep0[qualityMetric] * 100).toFixed(1);
                    const speedDiff = ((keep4.speed - keep0.speed) / keep0.speed * 100).toFixed(1);
                    const combinedDiff = ((keep4.combined - keep0.combined) / keep0.combined * 100).toFixed(1);
                    
                    const winner = parseFloat(combinedDiff) > 0 ? '--keep 4' : '--keep 0';
                    const qualityLabel = datasetType === 'longbench' ? 'ROUGE-L' : 'Accuracy';
                    
                    return `${winner} wins: ${qualityLabel} ${qualityDiff}%, Speed ${speedDiff}%, Combined ${combinedDiff}%`;
                  })()}
                </span>
              </div>
            )}
          </div>
        </>
      )}

      {/* StreamLLM Data Not Available Message */}
      {datasetType === 'longbench' && !filteredData.some(d => d.streamllm_enabled !== undefined && d.streamllm_enabled !== null && d.streamllm_enabled !== '') && (
        <div style={{ background: 'rgba(100, 116, 139, 0.1)', border: '1px solid rgba(100, 116, 139, 0.3)', borderRadius: 12, padding: 16, marginBottom: 24, marginTop: 32 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: '1.5rem' }}>ℹ️</span>
            <div>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#374151', marginBottom: 4 }}>StreamLLM Analysis Not Available</h3>
              <p style={{ fontSize: '0.85rem', color: '#cbd5e1', margin: 0 }}>
                This dataset doesn't include StreamLLM/attention sinks data. To see StreamLLM analysis, upload a CSV with 
                <code style={{ background: 'rgba(0,0,0,0.3)', padding: '2px 6px', borderRadius: 4, margin: '0 4px' }}>streamllm_enabled</code>,
                <code style={{ background: 'rgba(0,0,0,0.3)', padding: '2px 6px', borderRadius: 4, margin: '0 4px' }}>sink_count</code>, and
                <code style={{ background: 'rgba(0,0,0,0.3)', padding: '2px 6px', borderRadius: 4, margin: '0 4px' }}>sink_bias</code> columns.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* StreamLLM / Attention Sinks Analysis Section */}
      {datasetType === 'longbench' && filteredData.some(d => d.streamllm_enabled !== undefined && d.streamllm_enabled !== null && d.streamllm_enabled !== '') && (
        <>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 16, color: '#111827', borderBottom: '2px solid #e5e7eb', paddingBottom: 8, marginTop: 32 }}>🔮 StreamLLM & Attention Sinks Analysis</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: 20, marginBottom: 24 }}>
            {/* StreamLLM Enabled vs Disabled - ROUGE-L */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>StreamLLM Impact on ROUGE-L</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={streamllmStats} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="streamllm_enabled" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <YAxis stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={v => v?.toFixed(4)} />
                  <Legend />
                  <Bar dataKey="rougeL" name="ROUGE-L" fill="#7c3aed" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="rouge1" name="ROUGE-1" fill="#059669" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="rouge2" name="ROUGE-2" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* StreamLLM Impact on Speed */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>StreamLLM Impact on Speed</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={streamllmStats} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="streamllm_enabled" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <YAxis stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={v => v?.toFixed(1)} />
                  <Bar dataKey="speed" name="Speed (tok/s)" fill="#059669" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Sink Count Impact on ROUGE-L */}
            {sinkCountStats.length > 0 && (
              <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
                <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Sink Count Impact on ROUGE-L</h3>
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={sinkCountStats} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="sink_count" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                    <YAxis yAxisId="left" stroke="#8b5cf6" tick={{ fill: '#8b5cf6', fontSize: 11 }} />
                    <YAxis yAxisId="right" orientation="right" stroke="#10b981" tick={{ fill: '#10b981', fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} />
                    <Legend />
                    <Line yAxisId="left" type="monotone" dataKey="rougeL" name="ROUGE-L" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 5 }} />
                    <Line yAxisId="left" type="monotone" dataKey="rouge1" name="ROUGE-1" stroke="#ec4899" strokeWidth={1.5} dot={{ r: 4 }} strokeDasharray="5 5" />
                    <Line yAxisId="right" type="monotone" dataKey="speed" name="Speed (tok/s)" stroke="#10b981" strokeWidth={2} dot={{ r: 5 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Sink Count Impact on Speed */}
            {sinkCountStats.length > 0 && (
              <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
                <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Sink Count vs Combined Score</h3>
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={sinkCountStats} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="sink_count" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                    <YAxis stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} />
                    <Legend />
                    <Line type="monotone" dataKey="combined" name="Combined Score" stroke="#fbbf24" strokeWidth={2} dot={{ r: 5 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* StreamLLM Enabled: Speed vs ROUGE-L Scatter */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Speed vs ROUGE-L (StreamLLM Color-Coded)</h3>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" dataKey="decode_speed" name="Speed" unit=" tok/s" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <YAxis type="number" dataKey="rougeL" name="ROUGE-L" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip modelColors={modelColors} />} />
                  <Legend />
                  <Scatter name="StreamLLM Enabled" data={filteredData.filter(d => d.streamllm_enabled)} fill="#06b6d4">
                    {filteredData.filter(d => d.streamllm_enabled).map((entry, index) => <Cell key={`enabled-${index}`} fill="#06b6d4" fillOpacity={0.8} />)}
                  </Scatter>
                  <Scatter name="StreamLLM Disabled" data={filteredData.filter(d => !d.streamllm_enabled)} fill="#ef4444">
                    {filteredData.filter(d => !d.streamllm_enabled).map((entry, index) => <Cell key={`disabled-${index}`} fill="#ef4444" fillOpacity={0.8} />)}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
                      <ScatterLegend modelColors={modelColors} models={models} />
            </div>

            {/* Combined Score: StreamLLM On vs Off */}
            <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderRadius: 16, padding: 20 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12, color: '#374151' }}>Combined Score: StreamLLM On vs Off</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={streamllmStats} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="streamllm_enabled" stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} />
                  <YAxis stroke="#6b7280" tick={{ fill: '#374151', fontSize: 11 }} domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #d1d5db', borderRadius: 8 }} formatter={v => v?.toFixed(3)} />
                  <Bar dataKey="combined" name="Combined Score" fill="#fbbf24" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* StreamLLM Summary Stats */}
          <div style={{ background: '#ecfeff', border: '1px solid #a5f3fc', borderRadius: 16, padding: 20, marginBottom: 24 }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 12, color: '#06b6d4' }}>🔍 StreamLLM Impact Summary</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
              {streamllmStats.map((stat, idx) => {
                const enabled = stat.streamllm_enabled === 1 || stat.streamllm_enabled === true || stat.streamllm_enabled === 'on';
                return (
                  <div key={idx} style={{ background: enabled ? 'rgba(6, 182, 212, 0.15)' : 'rgba(239, 68, 68, 0.15)', border: `1px solid ${enabled ? 'rgba(6, 182, 212, 0.4)' : 'rgba(239, 68, 68, 0.4)'}`, borderRadius: 8, padding: 12 }}>
                    <div style={{ color: enabled ? '#06b6d4' : '#ef4444', fontSize: '0.9rem', fontWeight: 600, marginBottom: 8 }}>
                      {enabled ? '✓ Enabled' : '✗ Disabled'}
                    </div>
                    <div style={{ fontSize: '0.75rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 8px' }}>
                      <span style={{ color: '#374151' }}>ROUGE-L:</span><span style={{ fontWeight: 600 }}>{stat.rougeL?.toFixed(4) || 'N/A'}</span>
                      <span style={{ color: '#374151' }}>Speed:</span><span style={{ fontWeight: 600 }}>{stat.speed?.toFixed(1) || 'N/A'} tok/s</span>
                      <span style={{ color: '#374151' }}>Combined:</span><span style={{ fontWeight: 600 }}>{stat.combined?.toFixed(3) || 'N/A'}</span>
                      <span style={{ color: '#374151' }}>Runs:</span><span>{stat.count}</span>
                    </div>
                  </div>
                );
              })}
            </div>
            {streamllmStats.length === 2 && (
              <div style={{ marginTop: 16, padding: '12px', background: 'rgba(251, 191, 36, 0.1)', borderRadius: 8, border: '1px solid rgba(251, 191, 36, 0.3)', fontSize: '0.85rem' }}>
                <span style={{ color: '#fbbf24', fontWeight: 600 }}>💡 Verdict:</span>
                <span style={{ color: '#111827', marginLeft: 8 }}>
                  {(() => {
                    const enabled = streamllmStats.find(s => s.streamllm_enabled === 1 || s.streamllm_enabled === true || s.streamllm_enabled === 'on');
                    const disabled = streamllmStats.find(s => s.streamllm_enabled === 0 || s.streamllm_enabled === false || s.streamllm_enabled === 'off');
                    if (!enabled || !disabled) return 'Not enough data to compare';
                    
                    const rougeDiff = ((enabled.rougeL - disabled.rougeL) / disabled.rougeL * 100).toFixed(1);
                    const speedDiff = ((enabled.speed - disabled.speed) / disabled.speed * 100).toFixed(1);
                    const combinedDiff = ((enabled.combined - disabled.combined) / disabled.combined * 100).toFixed(1);
                    
                    return `StreamLLM ${parseFloat(combinedDiff) > 0 ? 'HELPS' : 'HURTS'}: ROUGE-L ${rougeDiff}%, Speed ${speedDiff}%, Combined ${combinedDiff}%`;
                  })()}
                </span>
              </div>
            )}
          </div>
        </>
      )}

      {/* Hyperparameter Impact Charts */}
      <h2 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: 16, color: '#111827' }}>📊 Hyperparameter Impact Analysis</h2>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: 20, marginBottom: 24 }}>
        {/* Sampling Parameters */}
        {tempStats.length > 0 && <ParamChart data={tempStats} param="temperature" title="Temperature Impact" datasetType={datasetType} />}
        {repPenStats.length > 0 && <ParamChart data={repPenStats} param="repeat_penalty" title="Repeat Penalty Impact" datasetType={datasetType} />}
        {topPStats.length > 0 && <ParamChart data={topPStats} param="top_p" title="Top-P Impact" datasetType={datasetType} />}
        {topKStats.length > 0 && <ParamChart data={topKStats} param="top_k" title="Top-K Impact" datasetType={datasetType} />}
        {minPStats.length > 0 && <ParamChart data={minPStats} param="min_p" title="Min-P Impact" datasetType={datasetType} />}
        {dryMultStats.length > 0 && <ParamChart data={dryMultStats} param="dry_multiplier" title="DRY Multiplier Impact" datasetType={datasetType} />}
        
        {/* Penalty Parameters */}
        {presencePenStats.length > 0 && <ParamChart data={presencePenStats} param="presence_penalty" title="Presence Penalty Impact" datasetType={datasetType} />}
        {freqPenStats.length > 0 && <ParamChart data={freqPenStats} param="frequency_penalty" title="Frequency Penalty Impact" datasetType={datasetType} />}
        
        {/* Context & Memory Parameters */}
        {ctxStats.length > 0 && <ParamChart data={ctxStats} param="ctx_size" title="Context Size Impact" datasetType={datasetType} />}
        {keepStats.length > 0 && <ParamChart data={keepStats} param="keep" title="Keep Parameter Impact (--keep)" datasetType={datasetType} />}
        {contextShiftStats.length > 0 && <ParamChart data={contextShiftStats} param="context_shift" title="Context Shift Impact" datasetType={datasetType} />}
        
        {/* Batch & Processing Parameters */}
        {batchStats.length > 0 && <ParamChart data={batchStats} param="batch_size" title="Batch Size Impact" datasetType={datasetType} />}
        {ubatchStats.length > 0 && <ParamChart data={ubatchStats} param="ubatch_size" title="UBatch Size Impact" datasetType={datasetType} />}
        {threadsStats.length > 0 && <ParamChart data={threadsStats} param="threads" title="Thread Count Impact" datasetType={datasetType} />}
        {nglStats.length > 0 && <ParamChart data={nglStats} param="ngl" title="GPU Layers (NGL) Impact" datasetType={datasetType} />}
        {pollStats.length > 0 && <ParamChart data={pollStats} param="poll_level" title="Poll Level Impact" datasetType={datasetType} />}
        
        {/* Dataset-Specific Parameters */}
        {datasetType === 'longbench' && tokenLimitStats.length > 0 && <ParamChart data={tokenLimitStats} param="token_limit" title="Token Limit Impact" datasetType={datasetType} />}
      </div>

      {/* Categorical Parameters */}
      <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 16, color: '#111827', borderBottom: '2px solid #e5e7eb', paddingBottom: 8 }}>🔧 Categorical Parameters</h2>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: 20, marginBottom: 24 }}>
        {flashStats.length > 0 && <BarParamChart data={flashStats} param="flash_attn" title="Flash Attention" />}
        {ctkStats.length > 0 && <BarParamChart data={ctkStats} param="ctk" title="KV Cache Key Type (CTK)" />}
        {ctvStats.length > 0 && <BarParamChart data={ctvStats} param="ctv" title="KV Cache Value Type (CTV)" />}
        {mmapStats.length > 0 && <BarParamChart data={mmapStats} param="use_mmap" title="Use Memory Mapping (mmap)" />}
        {splitModeStats.length > 0 && <BarParamChart data={splitModeStats} param="split_mode" title="Split Mode" />}
        {datasetType === 'longbench' && streamllmStats.length > 0 && <BarParamChart data={streamllmStats} param="streamllm_enabled" title="StreamLLM Enabled" />}
        {datasetType === 'longbench' && sinkCountStats.length > 0 && <BarParamChart data={sinkCountStats} param="sink_count" title="Sink Count (StreamLLM)" />}
      </div>

      {/* Summary Stats */}
      <div style={{ background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: 16, padding: 20, marginBottom: 24 }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: 12, color: '#059669' }}>📈 Summary Statistics</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12 }}>
          <div>
            <div style={{ color: '#6b7280', fontSize: '0.8rem' }}>Total Runs</div>
            <div style={{ fontSize: '1.3rem', fontWeight: 700 }}>{stats?.count}</div>
          </div>
          
          {datasetType === 'truthfulqa' && (
            <>
              <div>
                <div style={{ color: '#6b7280', fontSize: '0.8rem' }}>Best Accuracy</div>
                <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#8b5cf6' }}>{(stats?.maxAcc * 100).toFixed(1)}%</div>
              </div>
              {stats?.maxBleurt !== null && (
                <div>
                  <div style={{ color: '#6b7280', fontSize: '0.8rem' }}>Best BLEURT</div>
                  <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#06b6d4' }}>{stats?.maxBleurt.toFixed(3)}</div>
                </div>
              )}
            </>
          )}
          
          {datasetType === 'longbench' && (
            <>
              <div>
                <div style={{ color: '#6b7280', fontSize: '0.8rem' }}>Best ROUGE-L</div>
                <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#8b5cf6' }}>{stats?.maxRougeL.toFixed(4)}</div>
              </div>
              <div>
                <div style={{ color: '#6b7280', fontSize: '0.8rem' }}>Avg ROUGE-L</div>
                <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#8b5cf6' }}>{stats?.avgRougeL.toFixed(4)}</div>
              </div>
              <div>
                <div style={{ color: '#6b7280', fontSize: '0.8rem' }}>Best ROUGE-1</div>
                <div style={{ fontSize: '1.3rem', fontWeight: 700 }}>{stats?.maxRouge1.toFixed(4)}</div>
              </div>
            </>
          )}
          
          <div>
            <div style={{ color: '#6b7280', fontSize: '0.8rem' }}>Best Speed</div>
            <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#059669' }}>{stats?.maxSpeed.toFixed(1)} tok/s</div>
          </div>
          <div>
            <div style={{ color: '#6b7280', fontSize: '0.8rem' }}>Best Combined</div>
            <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#fbbf24' }}>{stats?.maxCombined.toFixed(3)}</div>
          </div>
          {stats?.avgRuntime !== null && (
            <div>
              <div style={{ color: '#6b7280', fontSize: '0.8rem' }}>Avg Runtime</div>
              <div style={{ fontSize: '1.3rem', fontWeight: 700 }}>{stats?.avgRuntime.toFixed(0)}s</div>
            </div>
          )}
        </div>

        <div style={{ marginTop: 12, padding: '10px 14px', background: `rgba(${datasetType === 'truthfulqa' ? '16, 185, 129' : '139, 92, 246'}, 0.1)`, borderRadius: 8, border: `1px solid rgba(${datasetType === 'truthfulqa' ? '16, 185, 129' : '139, 92, 246'}, 0.3)`, fontSize: '0.8rem' }}>
          <span style={{ color: datasetColor, fontWeight: 600 }}>ℹ️ Combined Score:</span>
          <span style={{ color: '#374151', marginLeft: 8 }}>
            {datasetType === 'truthfulqa' 
              ? hasBleurtData 
                ? '(norm_accuracy + norm_speed + norm_bleurt) / 3'
                : '(norm_accuracy + norm_speed) / 2'
              : '(norm_rougeL + norm_speed + norm_rouge1) / 3'}
          </span>
        </div>
      </div>

      <div style={{ textAlign: 'center', color: '#475569', fontSize: '0.8rem' }}>
        LLM Hyperparameter Analyzer • {filteredData.length} runs analyzed • {datasetLabel}
      </div>
    </div>
  );
}

export default App;
