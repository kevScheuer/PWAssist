{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :no-members:
   {% block classes %}
   {% if classes %}
   .. rubric:: Classes
   .. autosummary::
      :toctree:
      :template: autosummary/class.rst
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: Functions
   .. autosummary::
      :toctree:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}