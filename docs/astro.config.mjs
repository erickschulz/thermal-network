// docs/astro.config.mjs
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  integrations: [
    starlight({
      title: 'Thermal Network',
      logo: {
        src: './src/assets/thermal-network-logo.svg',
      },
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/erickschulz/thermal-network',
        },
      ],
      sidebar: [
        {
          label: 'Guide',
          items: [
            { label: 'Introduction', link: '/' },
            { label: 'Quickstart', link: '/guides/quickstart/' },
			{ label: 'Installation', link: '/guides/installation/' },
          ],
        },
        {
          label: 'API Reference',
          autogenerate: { directory: 'reference' },
        },
      ],
    }),
  ],
});