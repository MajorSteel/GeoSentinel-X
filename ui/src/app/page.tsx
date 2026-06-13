"use client";

import React, { useState } from 'react';
import DeckGL from '@deck.gl/react';
import { Map } from 'react-map-gl';
// import { GeoJsonLayer } from '@deck.gl/layers';

// Remember to add MAPBOX_ACCESS_TOKEN to your .env.local
const MAPBOX_ACCESS_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || 'pk.ey_placeholder';

const INITIAL_VIEW_STATE = {
  longitude: -122.12,
  latitude: 47.65,
  zoom: 10,
  pitch: 45,
  bearing: 0
};

export default function GeoSentinelMap() {
  const [layers, setLayers] = useState([]);

  const handleAnalyze = async () => {
    console.log("Analyzing current view with GeoSentinel-X...");
  };

  return (
    <div className="relative w-full h-screen bg-black overflow-hidden">
      <DeckGL
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}
        layers={layers}
      >
        <Map 
          mapboxAccessToken={MAPBOX_ACCESS_TOKEN} 
          mapStyle="mapbox://styles/mapbox/dark-v11" 
        />
      </DeckGL>

      {/* Control Panel Overlay */}
      <div className="absolute top-4 left-4 z-10 bg-slate-900/80 backdrop-blur-md p-6 rounded-xl border border-slate-700 shadow-2xl text-white w-80">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent mb-2">
          GeoSentinel-X
        </h1>
        <p className="text-sm text-slate-400 mb-6">Multi-Temporal Earth Intelligence</p>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1 text-slate-300">Analysis Task</label>
            <select className="w-full bg-slate-800 border border-slate-600 rounded-md p-2 text-white outline-none focus:ring-2 focus:ring-blue-500">
              <option>LULC Classification</option>
              <option>Change Detection</option>
              <option>Future Forecasting</option>
            </select>
          </div>
          
          <button 
            onClick={handleAnalyze}
            className="w-full py-2 bg-blue-600 hover:bg-blue-500 transition-colors rounded-md font-semibold mt-4 shadow-[0_0_15px_rgba(37,99,235,0.5)]"
          >
            Run Intelligence
          </button>
        </div>
      </div>
      
      {/* GeoLLM Chat Overlay */}
      <div className="absolute bottom-4 right-4 z-10 bg-slate-900/80 backdrop-blur-md p-4 rounded-xl border border-slate-700 shadow-2xl text-white w-96 h-80 flex flex-col">
        <h2 className="text-lg font-semibold mb-2 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          GeoLLM Copilot
        </h2>
        <div className="flex-1 bg-slate-800/50 rounded-lg p-3 text-sm text-slate-300 overflow-y-auto font-mono">
          [System]: Ready for natural language queries...
        </div>
        <input 
          type="text" 
          placeholder="Ask why vegetation is decreasing..." 
          className="mt-3 w-full bg-slate-800 border border-slate-600 rounded-md p-2 text-sm text-white outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
    </div>
  );
}
