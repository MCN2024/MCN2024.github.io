document.addEventListener('DOMContentLoaded', function() {
    const gifViewer = document.getElementById('gif-viewer');
    const nameDropdown = document.getElementById('name-dropdown');
    const backToTopButton = document.getElementById('back-to-top');
  
    // Fetch the GIF manifest
    fetch('/gif-manifest.json')
      .then(response => response.json())
      .then(data => {
        // Populate dropdown and create GIF elements
        data.gifs.forEach((gifName, index) => {
          // Add option to dropdown
          const option = document.createElement('option');
          option.value = gifName;
          option.textContent = gifName.replace('.gif', ''); // Remove .gif extension for display
          nameDropdown.appendChild(option);
  
          // Create GIF container
          const gifUrl = `/code/name_gifs/${gifName}`;
          const imgElement = document.createElement('img');
          imgElement.src = gifUrl;
          imgElement.alt = `${gifName} GIF`;
          const viewportWidth = window.innerWidth;
          imgElement.width = viewportWidth * 0.95;
          imgElement.classList.add('gif-image');
          
          const gifContainer = document.createElement('div');
          gifContainer.id = `gif-${gifName}`; // Add an ID for easy targeting
          gifContainer.classList.add('gif-container');
          gifContainer.appendChild(imgElement);
  
          // Add margin to create spacing
          gifContainer.style.marginBottom = '10px';
  
          // Add extra margin to the last GIF
          if (index === data.gifs.length - 1) {
            gifContainer.style.marginBottom = '20px';
          }
  
          gifViewer.appendChild(gifContainer);
        });
  
        // Add event listener to dropdown
        nameDropdown.addEventListener('change', function() {
          const selectedGif = this.value;
          if (selectedGif) {
            const targetElement = document.getElementById(`gif-${selectedGif}`);
            if (targetElement) {
              targetElement.scrollIntoView({ behavior: 'smooth' });
            }
          }
        });
      })
      .catch(error => console.error('Error fetching GIF manifest:', error));

    // Back to Top button functionality
    window.onscroll = function() {
        if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
        backToTopButton.style.display = "block";
        } else {
        backToTopButton.style.display = "none";
        }
    };

    // Back to Top button functionality
    backToTopButton.addEventListener('click', function(e) {
        e.preventDefault();
        window.scrollTo({
        top: 0,
        behavior: 'smooth'
        });
    });
  });
