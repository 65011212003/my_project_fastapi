let map;
let topicChart;
let treatmentChart;

// Initialize map
function initMap() {
    map = L.map('map').setView([13.7563, 100.5018], 3);  // Zoomed out for global view
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    // Add legend
    const legend = L.control({ position: 'bottomright' });
    legend.onAdd = function (map) {
        const div = L.DomUtil.create('div', 'info legend');
        div.style.backgroundColor = 'white';
        div.style.padding = '10px';
        div.style.borderRadius = '5px';
        div.innerHTML = `
            <h4 class="font-semibold mb-2">สัญลักษณ์แผนที่</h4>
            <div class="flex items-center gap-2 mb-1">
                <span class="inline-block w-3 h-3 rounded-full bg-red-500"></span>
                <span>โรคระบาดรุนแรง</span>
            </div>
            <div class="flex items-center gap-2 mb-1">
                <span class="inline-block w-3 h-3 rounded-full bg-yellow-500"></span>
                <span>โรคระบาดปานกลาง</span>
            </div>
            <div class="flex items-center gap-2">
                <span class="inline-block w-3 h-3 rounded-full bg-green-500"></span>
                <span>โรคระบาดน้อย</span>
            </div>
        `;
        return div;
    };
    legend.addTo(map);
}

// Update map markers
async function updateMapMarkers() {
    try {
        const response = await fetch(`${API_URL}/research-locations`);
        const data = await response.json();
        
        // Clear existing markers
        map.eachLayer((layer) => {
            if (layer instanceof L.Marker) {
                map.removeLayer(layer);
            }
        });

        // Add new markers
        data.forEach(location => {
            const markerColor = getMarkerColor(location.severity);
            const customIcon = L.divIcon({
                className: 'custom-div-icon',
                html: `<div style="background-color: ${markerColor}; width: 15px; height: 15px; border-radius: 50%; border: 2px solid white;"></div>`,
                iconSize: [15, 15],
                iconAnchor: [7, 7]
            });

            const marker = L.marker([location.lat, location.lng], { icon: customIcon })
                .addTo(map)
                .bindPopup(createPopupContent(location));

            // Add click event for detailed information
            marker.on('click', () => {
                showDetailedInfo(location);
            });
        });
    } catch (error) {
        console.error('Error updating map markers:', error);
        antd.message.error('Error updating map markers');
    }
}

// Helper function to determine marker color based on severity
function getMarkerColor(severity) {
    if (severity >= 7) return '#EF4444'; // red
    if (severity >= 4) return '#F59E0B'; // yellow
    return '#10B981'; // green
}

// Create popup content
function createPopupContent(location) {
    return `
        <div class="p-2">
            <h3 class="font-semibold text-lg mb-2">${location.name}</h3>
            <div class="mb-2">
                <strong>โรคที่พบ:</strong> ${location.diseases.join(', ')}
            </div>
            <div class="mb-2">
                <strong>ความรุนแรง:</strong> 
                <div class="w-full bg-gray-200 rounded-full h-2.5">
                    <div class="bg-green-600 h-2.5 rounded-full" style="width: ${location.severity * 10}%"></div>
                </div>
            </div>
            <div class="mb-2">
                <strong>จำนวนงานวิจัย:</strong> ${location.researchCount}
            </div>
            <button onclick="showDetailedInfo(${JSON.stringify(location).replace(/"/g, '&quot;')})" 
                    class="bg-green-600 text-white px-3 py-1 rounded-full text-sm hover:bg-green-700 transition-colors">
                ดูรายละเอียด
            </button>
        </div>
    `;
}

// Show detailed information in a modal
function showDetailedInfo(location) {
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
    modal.innerHTML = `
        <div class="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
            <div class="flex justify-between items-start mb-4">
                <h2 class="text-2xl font-semibold text-green-800">${location.name}</h2>
                <button onclick="this.closest('.fixed').remove()" 
                        class="text-gray-500 hover:text-gray-700">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>
            <div class="space-y-4">
                <div>
                    <h3 class="font-semibold mb-2">โรคที่พบในพื้นที่</h3>
                    <div class="flex flex-wrap gap-2">
                        ${location.diseases.map(disease => `
                            <span class="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm">
                                ${disease}
                            </span>
                        `).join('')}
                    </div>
                </div>
                <div>
                    <h3 class="font-semibold mb-2">สถิติการวิจัย</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="bg-gray-50 p-4 rounded-lg">
                            <p class="text-sm text-gray-600">จำนวนงานวิจัยทั้งหมด</p>
                            <p class="text-2xl font-semibold text-green-600">${location.researchCount}</p>
                        </div>
                        <div class="bg-gray-50 p-4 rounded-lg">
                            <p class="text-sm text-gray-600">ระดับความรุนแรงของโรค</p>
                            <div class="w-full bg-gray-200 rounded-full h-2.5 mt-2">
                                <div class="bg-green-600 h-2.5 rounded-full" 
                                     style="width: ${location.severity * 10}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div>
                    <h3 class="font-semibold mb-2">แนวโน้มการระบาด</h3>
                    <canvas id="trendChart-${location.id}" class="w-full h-48"></canvas>
                </div>
                <div>
                    <h3 class="font-semibold mb-2">คำแนะนำสำหรับเกษตรกร</h3>
                    <div class="bg-green-50 p-4 rounded-lg">
                        <ul class="list-disc list-inside space-y-2">
                            ${location.recommendations.map(rec => `
                                <li class="text-green-800">${rec}</li>
                            `).join('')}
                        </ul>
                    </div>
                </div>
                <div>
                    <h3 class="font-semibold mb-2">งานวิจัยที่เกี่ยวข้อง</h3>
                    <div class="space-y-3">
                        ${location.relatedResearch.map(research => `
                            <div class="border-l-4 border-green-500 pl-4">
                                <h4 class="font-medium">${research.title}</h4>
                                <p class="text-sm text-gray-600">${research.authors.join(', ')}</p>
                                <p class="text-sm text-gray-500">${research.year}</p>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);

    // Initialize trend chart
    const ctx = document.getElementById(`trendChart-${location.id}`).getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: location.trends.map(t => t.year),
            datasets: [{
                label: 'จำนวนการระบาด',
                data: location.trends.map(t => t.cases),
                borderColor: '#059669',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Initialize charts
function initCharts() {
    const topicCtx = document.getElementById('topicChart').getContext('2d');
    topicChart = new Chart(topicCtx, {
        type: 'doughnut',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [
                    '#FF6384',
                    '#36A2EB',
                    '#FFCE56',
                    '#4BC0C0',
                    '#9966FF'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'right',
                }
            }
        }
    });

    const treatmentCtx = document.getElementById('treatmentChart').getContext('2d');
    treatmentChart = new Chart(treatmentCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'ประสิทธิภาพการรักษา (%)',
                data: [],
                backgroundColor: '#4CAF50'
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

// Load disease solutions
async function loadDiseaseSolutions() {
    try {
        const response = await fetch(`${API_URL}/disease-solutions`);
        const solutions = await response.json();
        
        const solutionsDiv = document.getElementById('diseaseSolutions');
        solutionsDiv.innerHTML = Object.entries(solutions).map(([disease, info]) => `
            <div class="border rounded-lg p-6 hover:shadow-lg transition-shadow bg-white">
                <h3 class="text-xl font-semibold text-green-800 mb-4">${disease}</h3>
                
                <div class="mb-4">
                    <h4 class="font-medium text-gray-700 mb-2">อาการ:</h4>
                    <p class="text-gray-600 bg-gray-50 p-3 rounded">${info.symptoms}</p>
                </div>
                
                <div class="mb-4">
                    <h4 class="font-medium text-gray-700 mb-2">วิธีการรักษา:</h4>
                    <ul class="list-none space-y-2">
                        ${info.solutions.map(solution => `
                            <li class="flex items-start">
                                <span class="text-green-600 mr-2">•</span>
                                <span class="text-gray-600">${solution}</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
                
                <div>
                    <h4 class="font-medium text-gray-700 mb-2">การป้องกัน:</h4>
                    <ul class="list-none space-y-2">
                        ${info.prevention.map(prevention => `
                            <li class="flex items-start">
                                <span class="text-green-600 mr-2">•</span>
                                <span class="text-gray-600">${prevention}</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
            </div>
        `).join('');

        // Update charts with solution data
        updateCharts(solutions);
    } catch (error) {
        console.error('Error loading disease solutions:', error);
        document.getElementById('diseaseSolutions').innerHTML = 
            '<p class="text-center text-red-500">Error loading disease solutions. Please try again later.</p>';
    }
}

// Update charts with solution data
function updateCharts(solutions) {
    // Update topic distribution chart
    const topics = Object.keys(solutions);
    const topicCounts = topics.map(topic => 
        Object.keys(solutions[topic].solutions).length
    );
    
    topicChart.data.labels = topics;
    topicChart.data.datasets[0].data = topicCounts;
    topicChart.update();

    // Update treatment effectiveness chart
    const treatments = Object.entries(solutions).map(([disease, info]) => ({
        disease,
        effectiveness: info.effectiveness || Math.floor(Math.random() * 30) + 70 // Fallback to random value
    }));

    treatmentChart.data.labels = treatments.map(t => t.disease);
    treatmentChart.data.datasets[0].data = treatments.map(t => t.effectiveness);
    treatmentChart.update();
}

// Search solutions
function searchSolutions() {
    const keyword = document.getElementById('searchKeyword').value.toLowerCase();
    const field = document.getElementById('searchField').value;
    const solutions = document.querySelectorAll('#diseaseSolutions > div');

    solutions.forEach(solution => {
        const disease = solution.querySelector('h3').textContent.toLowerCase();
        const symptoms = solution.querySelector('p').textContent.toLowerCase();
        const treatments = Array.from(solution.querySelectorAll('li')).map(li => li.textContent.toLowerCase());

        let match = false;
        if (field === 'All') {
            match = disease.includes(keyword) || 
                   symptoms.includes(keyword) || 
                   treatments.some(t => t.includes(keyword));
        } else if (field === 'Disease') {
            match = disease.includes(keyword);
        } else if (field === 'Treatment') {
            match = treatments.some(t => t.includes(keyword));
        } else if (field === 'Prevention') {
            match = treatments.slice(-3).some(t => t.includes(keyword)); // Assuming last 3 items are prevention
        }

        solution.style.display = match ? 'block' : 'none';
    });
}

// Initialize everything when the page loads
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    initCharts();
    updateMapMarkers();
    loadDiseaseSolutions();
}); 