/**
 * Spatial Point-in-Polygon (Ray casting algorithm)
 */
export function isPointInPolygon(lat: number, lng: number, poly: [number, number][]): boolean {
  if (!poly || poly.length < 3) return true;
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [yi, xi] = poly[i];
    const [yj, xj] = poly[j];
    const intersect =
      yi > lat !== yj > lat && lng < ((xj - xi) * (lat - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

/**
 * Calculates approximate surface area of a polygon on earth in square kilometers (spherical excess)
 */
export function calculatePolygonAreaKm2(coords: [number, number][]): number {
  if (!coords || coords.length < 3) return 0;
  const radius = 6371; // Earth radius in km
  let total = 0;
  const len = coords.length;
  
  for (let i = 0; i < len; i++) {
    const [lat1, lon1] = coords[i];
    const [lat2, lon2] = coords[(i + 1) % len];
    
    const p1 = (lat1 * Math.PI) / 180;
    const p2 = (lat2 * Math.PI) / 180;
    const l1 = (lon1 * Math.PI) / 180;
    const l2 = (lon2 * Math.PI) / 180;
    
    total += (l2 - l1) * (2 + Math.sin(p1) + Math.sin(p2));
  }
  
  total = Math.abs((total * radius * radius) / 2);
  return total;
}
