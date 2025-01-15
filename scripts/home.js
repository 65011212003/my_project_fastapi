let diseaseChart, yearlyChart;

// Initialize charts
function initCharts() {
    const diseaseCtx = document.getElementById('diseaseChart').getContext('2d');
    diseaseChart = new Chart(diseaseCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Disease Occurrences',
                data: [],
                backgroundColor: '#4CAF50'
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

    const yearlyCtx = document.getElementById('yearlyChart').getContext('2d');
    yearlyChart = new Chart(yearlyCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Publications per Year',
                data: [],
                borderColor: '#2196F3',
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

// Load statistics
async function loadStatistics() {
    try {
        const response = await fetch(`${API_URL}/statistics`);
        const data = await response.json();

        // Update Disease Chart
        diseaseChart.data.labels = Object.keys(data.disease_counts);
        diseaseChart.data.datasets[0].data = Object.values(data.disease_counts);
        diseaseChart.update();

        // Update Yearly Chart
        const sortedYears = Object.entries(data.yearly_trends)
            .sort(([a], [b]) => a.localeCompare(b));
        yearlyChart.data.labels = sortedYears.map(([year]) => year);
        yearlyChart.data.datasets[0].data = sortedYears.map(([, count]) => count);
        yearlyChart.update();

        // Update Keyword Cloud
        const keywordCloud = document.getElementById('keywordCloud');
        keywordCloud.innerHTML = data.top_keywords
            .map(({word, count}) => `
                <span class="px-3 py-1 rounded-full text-sm font-medium"
                      style="background-color: rgba(76, 175, 80, ${Math.min(count / Math.max(...data.top_keywords.map(k => k.count)), 0.9)})">
                    ${word}
                </span>
            `)
            .join('');
    } catch (error) {
        console.error('Error loading statistics:', error);
        antd.message.error('Error loading statistics');
    }
}

// Load news
async function loadNews() {
    try {
        const response = await fetch(`${API_URL}/news`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const articles = await response.json();
        
        const container = document.getElementById('newsContainer');
        container.innerHTML = articles.map(article => `
            <div class="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow">
                ${article.urlToImage ? `
                    <img src="${article.urlToImage}" 
                         alt="${article.title}"
                         class="w-full h-48 object-cover"
                         onerror="this.onerror=null; this.src='https://via.placeholder.com/400x200?text=No+Image';">
                ` : `
                    <div class="w-full h-48 bg-gray-200 flex items-center justify-center">
                        <span class="text-gray-500">No Image Available</span>
                    </div>
                `}
                <div class="p-4">
                    <h3 class="text-lg font-semibold text-green-800 mb-2 line-clamp-2">
                        ${article.title}
                    </h3>
                    <p class="text-gray-600 text-sm mb-3 line-clamp-3">
                        ${article.description || 'No description available'}
                    </p>
                    <div class="flex justify-between items-center">
                        <span class="text-xs text-gray-500">
                            ${new Date(article.publishedAt).toLocaleDateString()}
                        </span>
                        <a href="${article.url}" 
                           target="_blank" 
                           class="text-green-600 hover:text-green-800 text-sm font-medium">
                            Read More →
                        </a>
                    </div>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading news:', error);
        document.getElementById('newsContainer').innerHTML = 
            '<p class="text-center text-red-500 col-span-full">Error loading news articles. Please try again later.</p>';
    }
}

// Check API Status
async function checkStatus() {
    try {
        const response = await fetch(`${API_URL}/status`);
        const status = await response.text();
        document.getElementById('status').textContent = status;
    } catch (error) {
        document.getElementById('status').textContent = 'API is not responding';
    }
}

// Initialize everything when the page loads
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    loadStatistics();
    loadNews();
    checkStatus();
}); 