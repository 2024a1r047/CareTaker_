// e:\model_test\caretaker\frontend\js\weatherWidget.js
class WeatherWidget {
    constructor(containerId, apiBaseUrl) {
        this.containerId = containerId;
        this.API_BASE_URL = apiBaseUrl || 'http://127.0.0.1:8000/api';
        this.TOKEN = localStorage.getItem('token');
        this.init();
    }

    // Initialize the widget
    init() {
        this.createWidget();
        this.updateTime();
        this.updateWeatherDisplay();

        // Update time every minute
        setInterval(() => this.updateTime(), 60000);

        // Update weather every 5 minutes
        setInterval(() => this.updateWeatherDisplay(), 300000);
    }

    // Create the widget HTML
    createWidget() {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="weather-widget" style="
                background: linear-gradient(135deg, rgba(87, 139, 135, 0.8), rgba(87, 139, 135, 0.9));
                border-radius: 16px;
                padding: 20px;
                border: 1px solid rgba(240, 231, 231, 0.1);
                margin-bottom: 20px;
            ">
                <div style="text-align: center; margin-bottom: 15px;">
                    <p id="current-time" style="margin: 4px 0 0; color: #e2e8f0; font-size: 0.9rem;">--:-- --</p>
                    <p id="weather-location" style="margin: 6px 0 0; color: #ffffff; font-size: 0.95rem; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 6px; min-height: 1.2em;">
                        <span style="color:#e2e8f0; font-weight:500;">Loading weather…</span>
                    </p>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center;">
                    <div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #ffffff;" id="tempValue">--°C</div>
                        <div style="font-size: 0.85rem; color: #cbd5e1;">Temperature</div>
                    </div>
                    <div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #ffffff;" id="humidityValue">--%</div>
                        <div style="font-size: 0.85rem; color: #cbd5e1;">Humidity</div>
                    </div>
                    <div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #ffffff;" id="aqiValue">--</div>
                        <div style="font-size: 0.85rem; color: #e2e8f0;">AQI <span id="aqiLabel" style="font-weight:600;"></span></div>
                    </div>
                </div>
            </div>
        `;
    }

    // Update time display
    updateTime() {
        const now = new Date();
        const timeElement = document.getElementById('current-time');
        if (timeElement) {
            timeElement.textContent = now.toLocaleString('en-US', {
                weekday: 'short',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        }
    }

    // Set current recipient and refresh
    setRecipient(recipientId) {
        this.recipientId = recipientId;
        this.updateWeatherDisplay();
    }

    // Fetch weather data
    async fetchWeatherData() {
        try {
            const url = this.recipientId
                ? `${this.API_BASE_URL}/weather/current?recipient_id=${this.recipientId}`
                : `${this.API_BASE_URL}/weather/current`;

            const response = await fetch(url, {
                headers: {
                    'Authorization': `Bearer ${this.TOKEN}`,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error('Failed to fetch weather data');
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching weather data:', error);
            return null;
        }
    }

    // Update weather display
    async updateWeatherDisplay() {
        const data = await this.fetchWeatherData();
        const locEl = document.getElementById('weather-location');
        if (!data || data.status === 'error') {
            console.log('No weather data available', data && data.error);
            if (locEl) {
                locEl.innerHTML = '<span style="color:#fde68a; font-weight:500;"><i class="fas fa-triangle-exclamation"></i> Weather unavailable right now</span>';
            }
            return;
        }

        if (locEl) {
            const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
            const icon = data.icon ? `<img src="https:${esc(data.icon)}" alt="" width="28" height="28" style="margin:-6px 0;">` : '';
            const place = [data.location, data.region].filter(Boolean).map(esc).join(', ');
            const feels = (data.feels_like !== undefined && data.feels_like !== null) ? ` · feels like ${esc(data.feels_like)}°C` : '';
            locEl.innerHTML = `${icon}<span><i class="fas fa-location-dot" style="opacity:0.8;"></i> ${place} · ${esc(data.condition || '')}${feels}</span>`;
        }
        const aqiLabelEl = document.getElementById('aqiLabel');
        if (aqiLabelEl) aqiLabelEl.textContent = data.aqi_label && data.aqi_label !== 'Unknown' ? `· ${data.aqi_label}` : '';

        // Weather data fetched


        // Update stats
        const updateElement = (id, value, suffix = '') => {
            const element = document.getElementById(id);
            if (element) {
                if (value === 'N/A' || value === undefined) {
                    element.textContent = '--' + suffix;
                } else {
                    element.textContent = value + suffix;
                }
            }
        };

        updateElement('tempValue', data.temperature, '°C');
        updateElement('humidityValue', data.humidity, '%');
        updateElement('aqiValue', data.aqi);
    }
}

// Export the WeatherWidget class
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WeatherWidget;
}