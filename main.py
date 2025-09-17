from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import folium
import json
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Ocean Hazard Hotspot Clustering API")

origins = [
     "https://sih-frontend-two.vercel.app"  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # allows only specified origins
    allow_credentials=True,
    allow_methods=["*"],         # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],         # allow all headers
)

# ---- Input Schema ----
class GeoPoint(BaseModel):
    longitude: float
    latitude: float

class GeoData(BaseModel):
    points: List[GeoPoint]

# ---- Clustering Function ----
def cluster_points(points: List[GeoPoint]):
    df = pd.DataFrame([p.dict() for p in points])

    # Convert degrees to radians for haversine distance
    coords = np.radians(df[["latitude", "longitude"]])
    kms_per_radian = 6371.0088
    epsilon = 50 / kms_per_radian  # 50 km neighborhood

    # Run DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree', metric='haversine').fit(coords)
    df["cluster"] = db.labels_

    # Save clusters JSON
    clusters_json = df.to_dict(orient="records")
    # with open("clusters_output.json", "w") as f:
    #     json.dump(clusters_json, f, indent=4)

    # Generate Folium map
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    colors = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue"]

    for _, row in df.iterrows():
        cluster_id = int(row["cluster"])
        color = colors[cluster_id % len(colors)] if cluster_id != -1 else "black"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            popup=f"Cluster {cluster_id}",
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    m.save("hotspot_map.html")
    return clusters_json

# ---- API Endpoints ----
@app.post("/cluster")
def create_clusters(data: GeoData):
    clusters = cluster_points(data.points)
    return {
        "message": "Clustering complete",
        "clusters_count": len(set([c["cluster"] for c in clusters])),
        "clusters": clusters,
        "map_file": "hotspot_map.html",
        "json_file": "clusters_output.json"
    }

@app.get("/")
def home():
    # return {"message": "Ocean Hazard Hotspot Clustering API is running!"}
    return FileResponse("hotspot_map.html", media_type="text/html")

# # ---- Run Server ----
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
