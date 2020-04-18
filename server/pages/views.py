import init_path

from django.shortcuts import render
from django.views.generic import TemplateView
from django.http import HttpResponse
from django.contrib.staticfiles.finders import find
from django.templatetags.static import static
from django.contrib.staticfiles.views import serve

from pages.infer_script import infer


class MenuPageView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'menu.html')


class TSNEPageView(TemplateView):
    def get(self, request, **kwargs):
        return serve(request, static('tsne.html'))


class DemoPageView(TemplateView):
    def get(self, request, **kwargs):
        svg = static('imgs/uspto_50k/9_r.svg')
        context = {
            'result': svg,
        }
        return render(request, 'demo.html', context)

    def post(self, request, **kwargs):
        if 'smi' in request.POST:
            smiles = request.POST['smi']
            print(f'**{smiles}**')
            svg = infer(smiles, 'cpu')
            with open('static/imgs/result.svg', 'w') as f:
                f.write(svg)
            svg = static('imgs/result.svg')
            context = {
                'result': svg,
            }
        return render(request, 'demo.html', context)
