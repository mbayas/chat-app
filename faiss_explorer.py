#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
from rich.prompt import Prompt, IntPrompt
import pickle

# Importar las librerías necesarias para FAISS
try:
    # Primero intentar las nuevas versiones de las librerías
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    print("Usando las últimas versiones de LangChain")
except ImportError:
    # Fallback a las versiones anteriores
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    print("Usando versiones anteriores de LangChain")

# Crear una consola rica para salida formateada
console = Console()

# Obtener la clave API de OpenAI
def get_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = Prompt.ask("[yellow]Ingresa tu API key de OpenAI[/yellow]")
    return api_key

# Directorio base para FAISS
BASE_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db')

def find_all_faiss_indices():
    """Encuentra todos los proyectos que contienen índices FAISS."""
    projects = []
    if not os.path.exists(BASE_DB_DIR):
        console.print(f"[bold red]El directorio de base de datos no existe: {BASE_DB_DIR}[/bold red]")
        return []
    
    for folder in os.listdir(BASE_DB_DIR):
        folder_path = os.path.join(BASE_DB_DIR, folder)
        if os.path.isdir(folder_path):
            faiss_file = os.path.join(folder_path, "index.faiss")
            pkl_file = os.path.join(folder_path, "index.pkl")
            
            if os.path.exists(faiss_file) and os.path.exists(pkl_file):
                console.print(f"[green]Índice FAISS encontrado en {folder}[/green]")
                projects.append(folder)
    
    return projects

def load_faiss_index(project_name, api_key):
    """Carga un índice FAISS para un proyecto específico."""
    project_db_dir = os.path.join(BASE_DB_DIR, project_name)
    
    if not os.path.exists(project_db_dir):
        console.print(f"[bold red]Error: No se encontró el directorio FAISS para el proyecto {project_name}[/bold red]")
        return None
    
    try:
        console.print(f"[bold blue]Cargando índice FAISS desde: {project_db_dir}[/bold blue]")
        
        # Inicializar OpenAI embeddings
        embeddings = OpenAIEmbeddings(api_key=api_key)
        
        # Cargar el índice FAISS
        db = FAISS.load_local(project_db_dir, embeddings, allow_dangerous_deserialization=True)
        console.print(f"[bold green]Índice FAISS cargado correctamente.[/bold green]")
        return db
    except Exception as e:
        console.print(f"[bold red]Error al cargar el índice FAISS: {str(e)}[/bold red]")
        import traceback
        traceback.print_exc()
        return None

def list_documents(db, limit=10):
    """Lista los documentos almacenados en el índice FAISS."""
    try:
        if hasattr(db, "docstore") and hasattr(db.docstore, "_dict"):
            documents = list(db.docstore._dict.values())
            
            total_docs = len(documents)
            console.print(f"[bold green]Total de documentos: {total_docs}[/bold green]")
            
            if not documents:
                console.print("[yellow]No se encontraron documentos en el índice.[/yellow]")
                return
            
            # Mostrar los primeros 'limit' documentos
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Índice", style="dim")
            table.add_column("Fuente", style="dim")
            table.add_column("Página", style="dim")
            table.add_column("Contenido (primeros 100 caracteres)")
            
            for i, doc in enumerate(documents[:limit]):
                source = doc.metadata.get('source', 'Desconocido')
                page = doc.metadata.get('page', 'N/A')
                content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                
                table.add_row(
                    str(i), 
                    source, 
                    str(page),
                    content
                )
            
            console.print(table)
            
            if total_docs > limit:
                console.print(f"[yellow]Mostrando {limit} de {total_docs} documentos. Use la búsqueda para ver más.[/yellow]")
        else:
            console.print("[yellow]No se pudo acceder a los documentos en el formato esperado.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error al listar documentos: {str(e)}[/bold red]")
        import traceback
        traceback.print_exc()

def view_document(db, index):
    """Ver un documento específico en detalle."""
    try:
        if hasattr(db, "docstore") and hasattr(db.docstore, "_dict"):
            documents = list(db.docstore._dict.values())
            
            if 0 <= index < len(documents):
                doc = documents[index]
                
                metadata_table = Table(show_header=True, header_style="bold blue")
                metadata_table.add_column("Clave")
                metadata_table.add_column("Valor")
                
                for key, value in doc.metadata.items():
                    metadata_table.add_row(str(key), str(value))
                
                console.print(Panel(f"[bold blue]Documento #{index}[/bold blue]"))
                console.print("[bold blue]Metadatos:[/bold blue]")
                console.print(metadata_table)
                console.print("[bold blue]Contenido:[/bold blue]")
                console.print(Panel(doc.page_content, width=100))
            else:
                console.print(f"[bold red]Error: Índice {index} fuera de rango. El rango válido es 0-{len(documents)-1}[/bold red]")
        else:
            console.print("[yellow]No se pudo acceder a los documentos en el formato esperado.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error al ver documento: {str(e)}[/bold red]")
        import traceback
        traceback.print_exc()

def search_documents(db, query, k=5):
    """Busca documentos similares a la consulta."""
    try:
        console.print(f"[bold blue]Buscando: '{query}'[/bold blue]")
        
        docs = db.similarity_search(query, k=k)
        
        console.print(f"[bold green]Se encontraron {len(docs)} documentos similares:[/bold green]")
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Desconocido')
            page = doc.metadata.get('page', 'N/A')
            
            console.print(Panel(
                f"[bold blue]Resultado #{i+1}[/bold blue]\n"
                f"[bold]Fuente:[/bold] {source}\n"
                f"[bold]Página:[/bold] {page}\n\n"
                f"{doc.page_content[:500]}..."
                if len(doc.page_content) > 500 else doc.page_content,
                width=100
            ))
    except Exception as e:
        console.print(f"[bold red]Error al buscar documentos: {str(e)}[/bold red]")
        import traceback
        traceback.print_exc()

def interactive_menu(api_key):
    """Menú interactivo para explorar contenidos de FAISS."""
    projects = find_all_faiss_indices()
    
    if not projects:
        console.print("[bold red]No se encontraron proyectos con índices FAISS.[/bold red]")
        console.print(f"[yellow]Buscando índices FAISS en: {BASE_DB_DIR}[/yellow]")
        return
    
    # Mostrar proyectos disponibles
    projects_table = Table(show_header=True, header_style="bold blue")
    projects_table.add_column("Índice")
    projects_table.add_column("Proyecto")
    
    for i, project in enumerate(projects):
        projects_table.add_row(str(i), project)
    
    console.print("[bold blue]Proyectos con índices FAISS:[/bold blue]")
    console.print(projects_table)
    
    # Seleccionar un proyecto
    project_index = IntPrompt.ask(
        "[bold green]Selecciona un proyecto por su índice[/bold green]",
        default=0,
        show_default=True
    )
    
    if project_index < 0 or project_index >= len(projects):
        console.print("[bold red]Índice de proyecto inválido.[/bold red]")
        return
    
    selected_project = projects[project_index]
    console.print(f"[bold green]Proyecto seleccionado: {selected_project}[/bold green]")
    
    # Cargar índice FAISS
    db = load_faiss_index(selected_project, api_key)
    if not db:
        return
    
    while True:
        console.print("\n" + "="*80)
        console.print(Panel(f"[bold blue]Explorador FAISS: {selected_project}[/bold blue]", width=80))
        console.print("[bold green]Opciones:[/bold green]")
        console.print("1. Listar documentos")
        console.print("2. Ver documento específico")
        console.print("3. Buscar documentos")
        console.print("4. Cambiar de proyecto")
        console.print("5. Salir")
        
        choice = Prompt.ask("[bold green]Selecciona una opción[/bold green]", choices=["1", "2", "3", "4", "5"])
        
        if choice == "1":
            limit = IntPrompt.ask("[bold green]Número de documentos a mostrar[/bold green]", default=10)
            list_documents(db, limit)
        
        elif choice == "2":
            index = IntPrompt.ask("[bold green]Índice del documento a ver[/bold green]", default=0)
            view_document(db, index)
        
        elif choice == "3":
            query = Prompt.ask("[bold green]Ingresa tu consulta de búsqueda[/bold green]")
            k = IntPrompt.ask("[bold green]Número de resultados[/bold green]", default=3)
            search_documents(db, query, k)
        
        elif choice == "4":
            return interactive_menu(api_key)
        
        elif choice == "5":
            console.print("[bold blue]¡Hasta luego![/bold blue]")
            break

if __name__ == "__main__":
    console.print(Panel.fit("[bold blue]FAISS Explorer[/bold blue]"))
    console.print(f"Buscando índices FAISS en: {BASE_DB_DIR}")
    
    # Obtener API key de OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    
    parser = argparse.ArgumentParser(description="Explorar contenidos de FAISS.")
    parser.add_argument("--list", action="store_true", help="Listar todos los proyectos con índices FAISS.")
    parser.add_argument("--project", type=str, help="Nombre del proyecto para explorar.")
    parser.add_argument("--search", type=str, help="Consulta para buscar en el índice FAISS.")
    
    args = parser.parse_args()
    
    if args.list:
        projects = find_all_faiss_indices()
        if projects:
            console.print("[bold blue]Proyectos con índices FAISS:[/bold blue]")
            for i, project in enumerate(projects):
                console.print(f"{i}: {project}")
    elif args.project and args.search:
        db = load_faiss_index(args.project, api_key)
        if db:
            search_documents(db, args.search)
    elif args.project:
        db = load_faiss_index(args.project, api_key)
        if db:
            list_documents(db)
    else:
        interactive_menu(api_key)
