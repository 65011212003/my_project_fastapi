// API Configuration
const API_URL = 'http://localhost:8000';
let currentLanguage = 'th';
const translationCache = {};

// Language Functions
async function translateText(text, targetLang) {
    if (!text || !text.trim()) return text;
    
    const cacheKey = `${text}_${targetLang}`;
    if (translationCache[cacheKey]) {
        return translationCache[cacheKey];
    }

    try {
        const response = await fetch(`${API_URL}/translate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                target_language: targetLang
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error('Translation error:', errorData);
            return text;
        }

        const data = await response.json();
        translationCache[cacheKey] = data.translated_text;
        return data.translated_text;
    } catch (error) {
        console.error('Translation error:', error);
        return text;
    }
}

async function translateBulk(texts, targetLang) {
    if (!texts || texts.length === 0) return texts;

    const validTexts = texts.filter(text => text && text.trim());
    if (validTexts.length === 0) return texts;

    try {
        const batchSize = 5;
        const results = [];
        
        for (let i = 0; i < validTexts.length; i += batchSize) {
            const batch = validTexts.slice(i, i + batchSize);
            try {
                const response = await fetch(`${API_URL}/translate-bulk`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        texts: batch,
                        target_language: targetLang
                    })
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    console.error('Batch translation error:', errorData);
                    results.push(...batch);
                    continue;
                }

                const data = await response.json();
                results.push(...data.translations.map(t => t.translated_text));
            } catch (error) {
                console.error('Batch translation error:', error);
                results.push(...batch);
            }
            
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        return results;
    } catch (error) {
        console.error('Bulk translation error:', error);
        return texts;
    }
}

async function changeLanguage(lang) {
    if (currentLanguage === lang) return;

    document.querySelectorAll('.language-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.getElementById(`lang-${lang}`).classList.add('active');

    try {
        const loadingMessage = document.createElement('div');
        loadingMessage.className = 'fixed top-4 right-4 bg-green-500 text-white px-4 py-2 rounded shadow-lg z-50';
        loadingMessage.textContent = 'Translating content...';
        document.body.appendChild(loadingMessage);

        const elements = document.querySelectorAll('h1, h2, h3, h4, h5, h6, p, span, button, a');
        const textsToTranslate = [];
        const elementsToUpdate = [];

        elements.forEach(element => {
            const text = element.textContent.trim();
            if (text && !element.closest('script') && !element.closest('style')) {
                textsToTranslate.push(text);
                elementsToUpdate.push(element);
            }
        });

        const batchSize = 5;
        for (let i = 0; i < textsToTranslate.length; i += batchSize) {
            const batch = textsToTranslate.slice(i, i + batchSize);
            try {
                const translations = await translateBulk(batch, lang);
                translations.forEach((translation, index) => {
                    const element = elementsToUpdate[i + index];
                    if (element) {
                        if (!element.getAttribute('data-original')) {
                            element.setAttribute('data-original', element.textContent);
                        }
                        element.textContent = translation;
                    }
                });
            } catch (error) {
                console.error('Translation error for batch:', error);
            }
            
            const progress = Math.min(100, Math.round((i + batchSize) / textsToTranslate.length * 100));
            loadingMessage.textContent = `Translating content... ${progress}%`;
            
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        currentLanguage = lang;
        document.documentElement.lang = lang;
        localStorage.setItem('preferredLanguage', lang);

        document.body.removeChild(loadingMessage);
        const successMessage = document.createElement('div');
        successMessage.className = 'fixed top-4 right-4 bg-green-500 text-white px-4 py-2 rounded shadow-lg z-50';
        successMessage.textContent = 'Language changed successfully';
        document.body.appendChild(successMessage);
        setTimeout(() => {
            document.body.removeChild(successMessage);
        }, 3000);

    } catch (error) {
        console.error('Language change error:', error);
        const errorMessage = document.createElement('div');
        errorMessage.className = 'fixed top-4 right-4 bg-red-500 text-white px-4 py-2 rounded shadow-lg z-50';
        errorMessage.textContent = 'Error changing language';
        document.body.appendChild(errorMessage);
        setTimeout(() => {
            document.body.removeChild(errorMessage);
        }, 3000);
    }
}

// Export Functions
async function exportData(format) {
    try {
        const response = await fetch(`${API_URL}/export/${format}`);
        if (!response.ok) throw new Error('Export failed');
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `rice_disease_analysis.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        antd.message.success(`Successfully exported data as ${format.toUpperCase()}`);
    } catch (error) {
        antd.message.error('Error exporting data');
    }
}

// Initialize language preference
document.addEventListener('DOMContentLoaded', () => {
    const savedLanguage = localStorage.getItem('preferredLanguage');
    if (savedLanguage && savedLanguage !== currentLanguage) {
        changeLanguage(savedLanguage);
    }
}); 