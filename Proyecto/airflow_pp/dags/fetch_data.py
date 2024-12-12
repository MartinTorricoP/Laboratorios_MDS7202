import requests
import os

def fetch_latest_files_from_gitlab(repo_url, branch_name, token, target_folder, limit=5):
    """
    Extrae los archivos más recientes de un repositorio de GitLab y los guarda localmente.

    Args:
        repo_url (str): URL del repositorio en GitLab.
        branch_name (str): Rama del repositorio desde donde extraer los archivos.
        token (str): Token de acceso personal de GitLab.
        target_folder (str): Carpeta local donde guardar los archivos extraídos.
        limit (int): Número de archivos más recientes a descargar.
    """
    # Configuración de la API
    api_url = f"{repo_url}/repository/tree"
    headers = {"PRIVATE-TOKEN": token}

    # Obtener la lista de archivos del repositorio
    response = requests.get(api_url, params={"ref": branch_name, "recursive": True}, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Error al obtener archivos: {response.status_code}, {response.text}")

    files = response.json()

    # Obtener la fecha del último commit para cada archivo
    files_with_dates = []
    for file_info in files:
        if file_info["type"] == "blob":  # Solo procesar archivos
            file_path = file_info["path"]
            commit_url = f"{repo_url}/repository/commits"
            commit_response = requests.get(
                commit_url,
                params={"path": file_path, "ref_name": branch_name},
                headers=headers
            )
            if commit_response.status_code == 200:
                commits = commit_response.json()
                if commits:
                    last_commit_date = commits[0]["committed_date"]
                    files_with_dates.append({"path": file_path, "date": last_commit_date})

    # Ordenar archivos por fecha 
    sorted_files = sorted(files_with_dates, key=lambda x: x["date"], reverse=True)

    # Descargar los archivos más recientes
    os.makedirs(target_folder, exist_ok=True)
    for file_info in sorted_files[:limit]:
        file_path = file_info["path"]
        file_url = f"{repo_url}/repository/files/{file_path.replace('/', '%2F')}/raw"
        file_name = os.path.basename(file_path)

        # Descargar el archivo
        file_response = requests.get(file_url, params={"ref": branch_name}, headers=headers)
        if file_response.status_code == 200:
            local_file_path = os.path.join(target_folder, file_name)
            with open(local_file_path, "wb") as f:
                f.write(file_response.content)
            print(f"Archivo descargado: {file_name} (Última modificación: {file_info['date']})")
        else:
            print(f"Error al descargar {file_name}: {file_response.status_code}, {file_response.text}")
