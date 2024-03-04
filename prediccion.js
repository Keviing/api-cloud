function predictDiabetes() {
    // Obtención de datos del formulario
    const data = {
        pregnancies: parseInt(document.getElementById('pregnancies').value),
        glucose: parseInt(document.getElementById('glucose').value),
        blood_pressure: parseInt(document.getElementById('blood_pressure').value),
        skin_thickness: parseInt(document.getElementById('skin_thickness').value),
        insulin: parseInt(document.getElementById('insulin').value),
        bmi: parseFloat(document.getElementById('bmi').value),
        diabetes_pedigree_function: parseFloat(document.getElementById('diabetes_pedigree_function').value),
        age: parseInt(document.getElementById('age').value),
    };
// Validar valores
if (Object.values(data).some(val => val < 0)) {
    alert('Por favor, ingrese valores mayores o iguales a 0 en todos los campos.');
    return;
}

// Validar que los campos esten completos
if (Object.values(data).some(val => isNaN(val))) {
    alert('Por favor, complete todos los campos.');
    return;
}
    // Enviar datos al servidor 
        fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(result => {
        const predictionResult = document.getElementById('predictionResult');
        predictionResult.innerText = ` ${result.has_diabetes ? "DIABETES: POSITIVO" : "DIABETES: NEGATIVO"}`;
    })
    .catch(error => console.error('Error:', error));
}

function clearForm() {
    // Limpiar todos los campos y el resultado de la predicción
    const form = document.getElementById('diabetesForm');
    form.reset();
    document.getElementById('predictionResult').innerText = '';
}


