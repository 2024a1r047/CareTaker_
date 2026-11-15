const urlParams = new URLSearchParams(window.location.search);
const role = urlParams.get('role');
const loginTitle = document.getElementById('loginTitle');

if (role) {
    if (role === 'doctor') {
        loginTitle.innerText = 'Doctor Login';
        const regLink = document.querySelector('.form-footer a');
        if (regLink) regLink.href = 'register.html?role=doctor';
    } else if (role === 'caretaker') {
        loginTitle.innerText = 'Caretaker Login';
        const regLink = document.querySelector('.form-footer a');
        if (regLink) regLink.href = 'register.html?role=caretaker';
    }
}

document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    try {
        console.log('Sending login data:', { username });
        const response = await fetch('http://localhost:8000/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                username,
                password
            })
        });
        
        console.log('Login response status:', response.status);

        const data = await response.json();

        if (response.ok) {
            console.log("FULL RESPONSE:", data);

            // Store the token and role
            localStorage.setItem('token', data.result.access_token);
            // a new login replaces any older login left in this tab (e.g. a caretaker's, from the
            // AI assistant) and the patient that account had selected
            try {
                sessionStorage.removeItem('token');
                sessionStorage.removeItem('selectedRecipientId');
                localStorage.removeItem('selectedRecipientId');
            } catch (e) { /* storage unavailable */ }
            const userRole = data?.result?.user?.role;

            if (!userRole) {
                alert("Role not found in response");
                console.error("Invalid response:", data);
                return;
            }

            console.log("User role:", userRole);
            localStorage.setItem('role', userRole);

            // Normalize role and redirect
            const roleNormalized = userRole.trim().toLowerCase();

            if (roleNormalized === 'doctor') {
                window.location.href = '/doctor_dashboard.html';
            } else if (roleNormalized === 'caretaker') {
                window.location.href = '/dashboard.html';
            } else {
                alert("Unknown role: " + userRole);
            }
        } else {
            alert(data.detail || 'Login failed. Please try again.');
        }
    } catch (error) {
        console.error('Error details:', error);
        alert('An error occurred. Please check the browser console for details.');
    }
});