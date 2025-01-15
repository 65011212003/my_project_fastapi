let currentPage = 1;
let totalPages = 1;
let pageSize = 10;

// Handle form submission
document.getElementById('queryForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = document.getElementById('query').value;
    const maxResults = document.getElementById('maxResults').value;

    try {
        const button = e.target.querySelector('button');
        button.disabled = true;
        button.textContent = 'Analyzing...';

        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query,
                max_results: parseInt(maxResults)
            })
        });
        const data = await response.json();
        antd.message.success(data.message);

        // Display cluster explanations if available
        if (data.cluster_explanations) {
            const clustersDiv = document.getElementById('clusters');
            clustersDiv.innerHTML = data.cluster_explanations.map(cluster => `
                <div class="border rounded-lg p-6 mb-4 hover:shadow-lg transition-shadow">
                    <h3 class="text-xl font-semibold text-green-800 mb-2">${cluster.title}</h3>
                    <div class="space-y-3">
                        <div>
                            <p class="font-medium text-gray-700">หัวข้อหลัก:</p>
                            <p class="text-gray-600">${cluster.main_focus}</p>
                        </div>
                        <div>
                            <p class="font-medium text-gray-700">คำอธิบาย:</p>
                            <p class="text-gray-600">${cluster.explanation}</p>
                        </div>
                        <div>
                            <p class="font-medium text-gray-700">การนำไปใช้:</p>
                            <p class="text-gray-600">${cluster.practical_use}</p>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        await loadClusters();
        await loadFarmerTopics();
        button.disabled = false;
        button.textContent = 'Analyze';
    } catch (error) {
        antd.message.error('Error starting analysis');
        const button = e.target.querySelector('button');
        button.disabled = false;
        button.textContent = 'Analyze';
    }
});

// Load clusters with visualization update
async function loadClusters() {
    try {
        const response = await fetch(`${API_URL}/clusters`);
        const clusters = await response.json();

        const clustersDiv = document.getElementById('clusters');
        clustersDiv.innerHTML = clusters.map(cluster => `
            <div class="border rounded p-4 hover:shadow-lg cursor-pointer transition-shadow" 
                 onclick="loadArticles(${cluster.cluster_number})">
                <h3 class="font-semibold">Cluster ${cluster.cluster_number}</h3>
                <p class="text-sm text-gray-600">Articles: ${cluster.article_count}</p>
                <div class="mt-2">
                    <p class="text-sm font-medium">Top Terms:</p>
                    <div class="flex flex-wrap gap-1 mt-1">
                        ${cluster.top_terms.map(term =>
                            `<span class="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">${term}</span>`
                        ).join('')}
                    </div>
                </div>
            </div>
        `).join('');
    } catch (error) {
        antd.message.error('Error loading clusters');
    }
}

// Load farmer topics
async function loadFarmerTopics() {
    try {
        const response = await fetch(`${API_URL}/farmer-topics`);
        const topics = await response.json();

        const topicsDiv = document.getElementById('farmerTopics');
        topicsDiv.innerHTML = topics.map(topic => `
            <div class="topic-card p-6">
                <div class="flex items-center gap-4 mb-4">
                    <div class="topic-icon">
                        ${getTopicIcon(topic.topic_id)}
                    </div>
                    <div>
                        <h3 class="font-semibold text-lg text-green-800">${topic.main_focus}</h3>
                        <div class="relevance-bar mt-2">
                            <div class="relevance-fill" style="width: ${topic.relevance_score}%"></div>
                        </div>
                        <p class="text-sm text-gray-600 mt-1">ความสำคัญ: ${topic.relevance_score}%</p>
                    </div>
                </div>
                <p class="text-gray-700 mb-4">${topic.simple_explanation}</p>
                <div class="flex flex-wrap gap-2">
                    ${topic.key_terms.map(term => `
                        <span class="key-term-tag">${term}</span>
                    `).join('')}
                </div>
            </div>
        `).join('');
    } catch (error) {
        antd.message.error('Error loading farmer-friendly topics');
    }
}

// Helper function to get topic icons
function getTopicIcon(topicId) {
    const icons = {
        0: '🛡️', // Prevention
        1: '🦠', // Disease and pathogens
        2: '🌾', // Rice management
        3: '🔍', // Disease inspection
        4: '🔬'  // Research and development
    };
    return icons[topicId] || '📋';
}

// Load articles with pagination
async function loadArticles() {
    try {
        const response = await fetch(`${API_URL}/articles?page=${currentPage}&page_size=${pageSize}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        // Update pagination info
        totalPages = data.total_pages;
        currentPage = data.page;
        
        // Update UI
        document.getElementById('pageInfo').textContent = `Page ${currentPage} of ${totalPages}`;
        document.getElementById('prevPage').disabled = currentPage <= 1;
        document.getElementById('nextPage').disabled = currentPage >= totalPages;
        
        // Display articles
        const container = document.getElementById('articlesContainer');
        container.innerHTML = data.items.map(article => `
            <div class="border-b border-gray-200 pb-6 mb-6 last:border-b-0">
                <h3 class="text-xl font-semibold text-green-800 mb-2">${article.Title}</h3>
                <p class="text-gray-600 mb-2">PMID: ${article.PMID} | Cluster: ${article.Cluster}</p>
                <p class="text-gray-700 mb-4">${article.Abstract}</p>
                <div class="flex flex-wrap gap-2">
                    ${Object.entries(article.Entities).map(([category, items]) => `
                        <div class="flex flex-col">
                            <span class="text-sm font-semibold text-green-700">${category}:</span>
                            <div class="flex flex-wrap gap-1">
                                ${items.map(item => `
                                    <span class="bg-green-100 text-green-800 text-xs px-2 py-1 rounded">${item}</span>
                                `).join('')}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading articles:', error);
        document.getElementById('articlesContainer').innerHTML = 
            '<p class="text-center text-red-500 my-4">Error loading articles. Please try again.</p>';
    }
}

// Navigation functions
function previousPage() {
    if (currentPage > 1) {
        currentPage--;
        loadArticles();
    }
}

function nextPage() {
    if (currentPage < totalPages) {
        currentPage++;
        loadArticles();
    }
}

function changePageSize() {
    pageSize = parseInt(document.getElementById('pageSize').value);
    currentPage = 1;  // Reset to first page
    loadArticles();
}

// Initialize everything when the page loads
document.addEventListener('DOMContentLoaded', () => {
    loadClusters();
    loadFarmerTopics();
    loadArticles();
}); 